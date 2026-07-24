// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow_array::{ArrayRef, DictionaryArray, Int32Array, new_empty_array};
use arrow_buffer::NullBuffer;
use arrow_schema::DataType as ArrowType;
use bytes::Bytes;

use crate::arrow::array_reader::primitive_array::{
    IntoBuffer, primitive_array_from_values,
};
use crate::arrow::array_reader::{ArrayReader, read_records, skip_records};
use crate::arrow::record_reader::GenericRecordReader;
use crate::arrow::record_reader::buffer::ValuesBuffer;
use crate::basic::Encoding;
use crate::column::page::PageIterator;
use crate::column::reader::decoder::{ColumnValueDecoder, ColumnValueDecoderImpl};
use crate::data_type::DataType;
use crate::encodings::decoding::{Decoder, PlainDecoder};
use crate::encodings::rle::RleDecoder;
use crate::errors::{ParquetError, Result};
use crate::schema::types::ColumnDescPtr;

enum PrimitiveDictionaryBuffer<T: DataType> {
    Dict {
        keys: Vec<i32>,
        values: Arc<Vec<T::T>>,
    },
    Values(Vec<T::T>),
}

impl<T> PrimitiveDictionaryBuffer<T>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
{
    fn as_keys(&mut self, dictionary: &Arc<Vec<T::T>>) -> Option<&mut Vec<i32>> {
        if matches!(self, Self::Values(values) if values.is_empty()) {
            *self = Self::Dict {
                keys: Vec::new(),
                values: Arc::clone(dictionary),
            };
        }

        match self {
            Self::Dict { keys, values } => {
                if Arc::ptr_eq(values, dictionary) {
                    Some(keys)
                } else if keys.is_empty() {
                    *values = Arc::clone(dictionary);
                    Some(keys)
                } else {
                    None
                }
            }
            Self::Values(_) => None,
        }
    }

    fn spill_values(&mut self) -> &mut Vec<T::T> {
        if let Self::Dict { keys, values } = self {
            let expanded = keys.iter().map(|key| values[*key as usize]).collect();
            *self = Self::Values(expanded);
        }
        match self {
            Self::Values(values) => values,
            _ => unreachable!(),
        }
    }
}

impl<T> ValuesBuffer for PrimitiveDictionaryBuffer<T>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
{
    fn with_capacity(capacity: usize) -> Self {
        Self::Values(Vec::with_capacity(capacity))
    }

    fn pad_nulls(
        &mut self,
        read_offset: usize,
        values_read: usize,
        levels_read: usize,
        valid_mask: &[u8],
    ) {
        match self {
            Self::Dict { keys, .. } => {
                keys.pad_nulls(read_offset, values_read, levels_read, valid_mask)
            }
            Self::Values(values) => {
                values.pad_nulls(read_offset, values_read, levels_read, valid_mask)
            }
        }
    }
}

enum MaybeDictionaryDecoder<T: DataType> {
    Dict {
        decoder: RleDecoder,
        max_remaining_values: usize,
    },
    Fallback(ColumnValueDecoderImpl<T>),
}

struct PrimitiveDictionaryDecoder<T: DataType> {
    column_desc: ColumnDescPtr,
    dictionary: Option<Arc<Vec<T::T>>>,
    decoder: Option<MaybeDictionaryDecoder<T>>,
}

impl<T> ColumnValueDecoder for PrimitiveDictionaryDecoder<T>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
{
    type Buffer = PrimitiveDictionaryBuffer<T>;

    fn new(column_desc: &ColumnDescPtr) -> Self {
        Self {
            column_desc: Arc::clone(column_desc),
            dictionary: None,
            decoder: None,
        }
    }

    fn set_dict(
        &mut self,
        buf: Bytes,
        num_values: u32,
        encoding: Encoding,
        _is_sorted: bool,
    ) -> Result<()> {
        if !matches!(
            encoding,
            Encoding::PLAIN | Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
        ) {
            return Err(nyi_err!(
                "Invalid/Unsupported encoding type for dictionary: {}",
                encoding
            ));
        }

        let mut decoder = PlainDecoder::<T>::new(self.column_desc.type_length());
        decoder.set_data(buf, num_values as usize)?;
        let mut values = vec![T::T::default(); num_values as usize];
        let decoded = decoder.get(&mut values)?;
        values.truncate(decoded);
        self.dictionary = Some(Arc::new(values));
        Ok(())
    }

    fn set_data(
        &mut self,
        encoding: Encoding,
        data: Bytes,
        num_levels: usize,
        num_values: Option<usize>,
    ) -> Result<()> {
        self.decoder = Some(match encoding {
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                let bit_width = data[0];
                let mut decoder = RleDecoder::new(bit_width);
                decoder.set_data(data.slice(1..))?;
                MaybeDictionaryDecoder::Dict {
                    decoder,
                    max_remaining_values: num_values.unwrap_or(num_levels),
                }
            }
            _ => {
                let mut decoder = ColumnValueDecoderImpl::<T>::new(&self.column_desc);
                decoder.set_data(encoding, data, num_levels, num_values)?;
                MaybeDictionaryDecoder::Fallback(decoder)
            }
        });
        Ok(())
    }

    fn read(&mut self, out: &mut Self::Buffer, num_values: usize) -> Result<usize> {
        match self.decoder.as_mut().expect("decoder set") {
            MaybeDictionaryDecoder::Dict {
                decoder,
                max_remaining_values,
            } => {
                let len = num_values.min(*max_remaining_values);
                let dictionary = self
                    .dictionary
                    .as_ref()
                    .ok_or_else(|| general_err!("missing dictionary page for column"))?;
                let keys = match out.as_keys(dictionary) {
                    Some(keys) => keys,
                    None => {
                        let values = out.spill_values();
                        let mut keys = vec![0_i32; len];
                        let read = decoder.get_batch(&mut keys)?;
                        values.extend(
                            keys[..read]
                                .iter()
                                .map(|key| dictionary[*key as usize]),
                        );
                        *max_remaining_values -= read;
                        return Ok(read);
                    }
                };
                let start = keys.len();
                keys.resize(start + len, 0);
                let read = decoder.get_batch(&mut keys[start..])?;
                keys.truncate(start + read);
                *max_remaining_values -= read;
                Ok(read)
            }
            MaybeDictionaryDecoder::Fallback(decoder) => {
                decoder.read(out.spill_values(), num_values)
            }
        }
    }

    fn skip_values(&mut self, num_values: usize) -> Result<usize> {
        match self.decoder.as_mut().expect("decoder set") {
            MaybeDictionaryDecoder::Dict {
                decoder,
                max_remaining_values,
            } => {
                let len = num_values.min(*max_remaining_values);
                let skipped = decoder.skip(len)?;
                *max_remaining_values -= skipped;
                Ok(skipped)
            }
            MaybeDictionaryDecoder::Fallback(decoder) => decoder.skip_values(num_values),
        }
    }
}

struct PrimitiveDictionaryReader<T>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
{
    data_type: ArrowType,
    pages: Box<dyn PageIterator>,
    def_levels_buffer: Option<Vec<i16>>,
    rep_levels_buffer: Option<Vec<i16>>,
    record_reader:
        GenericRecordReader<PrimitiveDictionaryBuffer<T>, PrimitiveDictionaryDecoder<T>>,
    _marker: PhantomData<T>,
}

impl<T> ArrayReader for PrimitiveDictionaryReader<T>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
    Vec<T::T>: IntoBuffer,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn read_records(&mut self, batch_size: usize) -> Result<usize> {
        read_records(&mut self.record_reader, self.pages.as_mut(), batch_size)
    }

    fn consume_batch(&mut self) -> Result<ArrayRef> {
        self.def_levels_buffer = self.record_reader.consume_def_levels();
        self.rep_levels_buffer = self.record_reader.consume_rep_levels();

        if self.record_reader.num_values() == 0 {
            return Ok(new_empty_array(&self.data_type));
        }

        let nulls = self
            .record_reader
            .consume_bitmap_buffer()
            .and_then(|buffer| {
                NullBuffer::from_unsliced_buffer(buffer, self.record_reader.num_values())
            });
        let buffer = self.record_reader.consume_record_data();
        let ArrowType::Dictionary(_, value_type) = &self.data_type else {
            unreachable!()
        };
        let array: ArrayRef = match buffer {
            PrimitiveDictionaryBuffer::Dict { keys, values } => {
                let values =
                    primitive_array_from_values::<T>((*values).clone(), value_type, None)?;
                Arc::new(DictionaryArray::try_new(
                    Int32Array::new(keys.into(), nulls),
                    values,
                )?)
            }
            PrimitiveDictionaryBuffer::Values(values) => {
                let values = primitive_array_from_values::<T>(values, value_type, nulls)?;
                crate::arrow::array_reader::primitive_array::pack_dictionary(
                    &ArrowType::Int32,
                    values.as_ref(),
                )?
            }
        };
        self.record_reader.reset();
        Ok(array)
    }

    fn skip_records(&mut self, num_records: usize) -> Result<usize> {
        skip_records(&mut self.record_reader, self.pages.as_mut(), num_records)
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.def_levels_buffer.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.rep_levels_buffer.as_deref()
    }
}

pub fn make_primitive_dictionary_reader<T>(
    pages: Box<dyn PageIterator>,
    column_desc: ColumnDescPtr,
    data_type: ArrowType,
    batch_size: usize,
) -> Result<Box<dyn ArrayReader>>
where
    T: DataType,
    T::T: Copy + Default + Send + Sync,
    Vec<T::T>: IntoBuffer,
{
    if !matches!(data_type, ArrowType::Dictionary(_, _)) {
        return Err(general_err!(
            "invalid non-dictionary data type for primitive dictionary reader - {}",
            data_type
        ));
    }
    Ok(Box::new(PrimitiveDictionaryReader::<T> {
        data_type,
        pages,
        def_levels_buffer: None,
        rep_levels_buffer: None,
        record_reader: GenericRecordReader::new(column_desc, batch_size),
        _marker: PhantomData,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::compute::cast;
    use arrow_array::{Array, BooleanArray, Int64Array, RecordBatch};
    use arrow_array::cast::AsArray;
    use arrow_schema::{ArrowError, Field, Schema};

    use crate::arrow::ArrowWriter;
    use crate::arrow::ProjectionMask;
    use crate::arrow::arrow_reader::{
        ArrowPredicate, ParquetRecordBatchReaderBuilder, RowFilter,
    };
    use crate::file::properties::WriterProperties;

    struct DictionaryPredicate {
        projection: ProjectionMask,
    }

    impl ArrowPredicate for DictionaryPredicate {
        fn projection(&self) -> &ProjectionMask {
            &self.projection
        }

        fn preserve_primitive_dictionaries(&self) -> bool {
            true
        }

        fn evaluate(&mut self, batch: RecordBatch) -> std::result::Result<BooleanArray, ArrowError> {
            assert!(matches!(
                batch.column(0).data_type(),
                ArrowType::Dictionary(key, value)
                    if key.as_ref() == &ArrowType::Int32
                        && value.as_ref() == &ArrowType::Int64
            ));
            let values = cast(batch.column(0), &ArrowType::Int64)?;
            let values = values.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(values.iter().map(|value| value.map(|value| value == 2)).collect())
        }
    }

    #[test]
    fn preserves_primitive_dictionary_for_predicate_only() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            ArrowType::Int64,
            true,
        )]));
        let values = Arc::new(Int64Array::from(vec![
            Some(1),
            Some(2),
            None,
            Some(3),
            Some(2),
            Some(1),
        ]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).unwrap();
        let properties = WriterProperties::builder()
            .set_dictionary_enabled(true)
            .build();
        let mut parquet = Vec::new();
        let mut writer =
            ArrowWriter::try_new(&mut parquet, Arc::clone(&schema), Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(parquet)).unwrap();
        let projection = ProjectionMask::leaves(builder.parquet_schema(), [0]);
        let filter = RowFilter::new(vec![Box::new(DictionaryPredicate { projection })]);
        let output: Vec<RecordBatch> = builder
            .with_row_filter(filter)
            .build()
            .unwrap()
            .collect::<std::result::Result<Vec<_>, ArrowError>>()
            .unwrap();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0].schema(), schema);
        assert_eq!(
            output[0].column(0).as_primitive::<arrow_array::types::Int64Type>(),
            &Int64Array::from(vec![2, 2])
        );
    }
}
