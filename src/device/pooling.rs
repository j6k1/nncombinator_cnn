use rayon::prelude::{ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use nncombinator::arr::{Arr, SerializedVec, SerializedVecView};
use nncombinator::device::DeviceCpu;
use nncombinator::error::{EvaluateError, SizeMismatchError, TrainingError};
use nncombinator::mem::AsRawSlice;
use nncombinator::ope::UnitValue;
use crate::{Assert, assert_convolution, IsTrue};
use crate::collection::{expand_image, Image, Images, ImagesView, reduce_images};

pub trait DeviceMaxPooling2D<U,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where U: UnitValue<U> {

    /// Forward propagation calculation
    /// # Arguments
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_maxpooling_2d<'a>(&self, input:ImagesView<'a,U,C,H,W>)
                             -> Result<Images<U,C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_maxpooling_2d<'a>(&self, loss:ImagesView<'a,U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                              input:ImagesView<'a,U,C,H,W>,
                              output:ImagesView<'a,U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>)
                              -> Result<Images<U,C,H,W>, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    ///* [`EvaluateError`]
    fn batch_forward_maxpooling_2d<'a>(&self, input:SerializedVecView<'a,U,Images<U,C,H,W>>)
                                   -> Result<SerializedVec<U,Images<U,C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>, EvaluateError>;
    /// Error back propagation calculation in batch
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_maxpooling_2d<'a>(&self,loss:SerializedVecView<'a,U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                    input:SerializedVecView<'a,U,Images<U,C,H,W>>,
                                    output:SerializedVecView<'a,U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>)
                                    -> Result<SerializedVec<U,Images<U,C,H,W>>, TrainingError>;
}
impl<U,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> for DeviceCpu<U>
    where U: UnitValue<U>,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn forward_maxpooling_2d<'a>(&self, input: ImagesView<'a,U, C, H, W>)
        -> Result<Images<U,C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError> {
        Ok(input.iter().map(|i| {
            expand_image::<U, H, W, FH, FW, PAD, S>(i)?.into_iter().map(|i| {
                i.into_iter().map(|i| {
                    i.as_raw_slice().iter().fold(U::initial_max_value(), |acc, i| {
                        acc.max(i)
                    })
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U, { (W + 2 * PAD - FW) / S + 1 }>>, SizeMismatchError>>()?.try_into()
        }).collect::<Result<Vec<Image<U,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()?)
    }

    fn backward_maxpooling_2d<'a>(&self, loss: ImagesView<'a,U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                              input: ImagesView<'a,U, C, H, W>,
                              _: ImagesView<'a,U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>) -> Result<Images<U, C, H, W>, TrainingError> {
        Ok(input.iter().map(|i| {
            Ok(expand_image::<U, H, W, FH, FW, PAD, S>(i)?.into_iter().map(|i| {
                i.into_iter().map(|i| {
                    let ((y,x),_) = i.iter().enumerate().fold(((0,0), U::initial_max_value()), | ((y,x),max), (cy,i)| {
                        i.iter().enumerate().fold(((y,x),max), |((y,x),max), (cx,&i)| {
                            if !(i <= max) {
                                ((cy,cx),i)
                            } else {
                                ((x,y),max)
                            }
                        })
                    });

                    let mut r = Image::new();

                    r[(y,x)] = U::one();

                    r
                }).collect::<Vec<Image<U,FH,FW>>>()
            }).collect::<Vec<Vec<Image<U,FH,FW>>>>())
        }).zip(loss.iter()).map(|(f,l)| {
            reduce_images::<U,H,W,FH,FW,PAD,S>(f?.iter().zip(l.iter()).map(|(f,l)| {
                Ok(f.iter().zip(l.iter()).map(|(f,&l)| {
                    f.as_raw_slice().iter().map(|&f| {
                        f * l
                    }).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Image<U,FH,FW>>,SizeMismatchError>>()?)
            }).collect::<Result<Vec<Vec<Image<U,FH,FW>>>,SizeMismatchError>>()?)
        }).collect::<Result<Vec<Image<U,H,W>>,SizeMismatchError>>()?.try_into()?)
    }

    fn batch_forward_maxpooling_2d<'a>(&self, input: SerializedVecView<'a,U, Images<U, C, H, W>>)
        -> Result<SerializedVec<U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>, EvaluateError> {
        Ok(input.par_iter().map(|i| {
            Ok(self.forward_maxpooling_2d(i)?)
        }).collect::<Result<Vec<Images<U,C,{ (H + 2 * PAD - FH) / S + 1 }, { (W + 2 * PAD - FW) / S + 1 }>>,EvaluateError>>()?.into())
    }

    fn batch_backward_maxpooling_2d<'a>(&self, loss: SerializedVecView<'a,U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                    input: SerializedVecView<'a,U, Images<U, C, H, W>>,
                                    output: SerializedVecView<'a,U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>) -> Result<SerializedVec<U, Images<U, C, H, W>>, TrainingError> {
        Ok(input.par_iter().zip(loss.par_iter()).zip(output.par_iter()).map(|((i,l),o)| {
            Ok(self.backward_maxpooling_2d(l,i,o)?)
        }).collect::<Result<Vec<Images<U,C,H,W>>,TrainingError>>()?.into())
    }
}