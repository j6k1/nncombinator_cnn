use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use nncombinator::arr::{Arr, Arr2, Arr3, Arr4, SerializedVec, SerializedVecView};
use nncombinator::device::DeviceCpu;
use nncombinator::error::{EvaluateError, TrainingError};
use nncombinator::error::SizeMismatchError;
use nncombinator::ope::UnitValue;
use crate::{Assert, assert_convolution, IsTrue};

use crate::collection::{expand_image, Image, Images, ImagesView, reduce_images};

/// Trait that defines the implementation of various calculation processes in the convolution layer
pub trait DeviceConvolution<U,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where U: UnitValue<U> {
    /// Forward propagation calculation
    /// # Arguments
    /// * `input` - Input values from upper layers
    /// * `kernel` - filter weights
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`EvaluateError`]
    fn forward_convolution<'a>(&self, input:ImagesView<'a,U,C,H,W>, kernel:&F)
        -> Result<Images<U,K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError>;
    /// Error back propagation calculation
    /// # Arguments
    /// * `loss` - loss
    /// * `kernel` - filter weights
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_convolution<'a>(&self, loss:ImagesView<'a,U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                            kernel:&F,)
        -> Result<Images<U,C,H,W>, TrainingError>;
    /// Calculate the gradient of the weights
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn backward_weight_gradient_convolution<'a>(&self,
                                loss: ImagesView<'a,U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                input: ImagesView<'a,U,C,H,W>)
        -> Result<Arr4<U,K,C,FH,FW>, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `input` - Input values from upper layers
    /// * `kernel` - filter weights
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    ///* [`EvaluateError`]
    fn batch_forward_convolution<'a>(&self, input:SerializedVecView<'a,U,Images<U,C,H,W>>, kernel:&F)
        -> Result<SerializedVec<U,Images<U,K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>, EvaluateError>;
    /// Error back propagation calculation in batch
    /// # Arguments
    /// * `loss` - loss
    /// * `kernel` - filter weights
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_convolution<'a>(&self,loss:SerializedVecView<'a,U,Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                  kernel:&F)
       -> Result<SerializedVec<U,Images<U,C,H,W>>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient_convolution<'a>(&self,
                                loss: SerializedVecView<'a,U,Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                input: SerializedVecView<'a,U,Images<U,C,H,W>>)
        -> Result<Arr4<U,K,C,FH,FW>, TrainingError>;
}
impl<U,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> DeviceConvolution<U,Arr4<U,K,C,FH,FW>,C,K,H,W,FH,FW,PAD,S> for DeviceCpu<U>
    where U: UnitValue<U>,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn forward_convolution<'a>(&self, input: ImagesView<'a,U, C, H, W>, kernel: &Arr4<U,K,C,FH,FW>)
        -> Result<Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError> {
        Ok(kernel.par_iter().map(|k| {
            k.par_iter().zip(input.par_iter()).map(|(k,i)| {
                expand_image::<U,H,W,FH,FW,PAD,S>(i)?.into_par_iter().map(|i| {
                    i.into_par_iter().map(|i| {
                        i.iter().zip(k.iter()).map(|(i, k)| {
                            i.iter().zip(k.iter()).map(|(&i, &k)| i * k).collect::<Vec<U>>()
                        }).fold(U::default(), | acc, i | {
                            acc + i.iter().fold(U::default(), | acc, &i | acc + i)
                        })
                    }).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()
            }).reduce(|| Ok(Image::new()), | acc, i | {
                acc.and_then(|acc| i.and_then(|i| {
                   acc.par_iter().zip(i.par_iter()).map(|(acc,i)| {
                       acc.par_iter().zip(i.par_iter()).map(|(&acc,&p)| acc + p).collect::<Vec<U>>().try_into()
                   }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()
                }))
            })
        }).collect::<Result<Vec<Image<U,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()?)
    }

    fn backward_convolution<'a>(&self, loss: ImagesView<'a,U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, kernel: &Arr4<U,K,C,FH,FW>)
        -> Result<Images<U,C,H,W>, TrainingError> {
        Ok(loss.par_iter().zip(kernel.par_iter()).map(|(l,k)| {
            Ok(k.par_iter().map(|k| {
               reduce_images::<U,H,W,FH,FW,PAD,S>(l.par_iter().map(|l| {
                    l.par_iter().map(|&l| {
                        k.par_iter().map(|k| {
                            k.par_iter().map(|&k| k * l).collect::<Vec<U>>().try_into()
                        }).collect::<Result<Vec<Arr<U, FW>>, SizeMismatchError>>()?.try_into()
                    }).collect::<Result<Vec<Image<U, FH, FW>>, SizeMismatchError>>()
                }).collect::<Result<Vec<Vec<Image<U, FH, FW>>>,SizeMismatchError>>()?)
            }).collect::<Result<Vec<Image<U,H,W>>,SizeMismatchError>>()?)
        }).fold(|| Ok(Images::new()), | acc, i | {
            acc.and_then(|acc| i.and_then(|i| acc.par_iter().zip(i.into_par_iter()).map(|(acc,i)| {
                acc.par_iter().zip(i.par_iter()).map(|(acc,i)| {
                    acc.par_iter().zip(i.par_iter()).map(|(&acc,&i)| acc + i).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Arr<U,W>>,SizeMismatchError>>()?.try_into()
            }).collect::<Result<Vec<Image<U,H,W>>,SizeMismatchError>>()))?.try_into()
        }).reduce(|| Ok(Images::new()), | acc, i | {
            acc.and_then(|acc| i.and_then(|i| {
                acc.par_iter().zip(i.par_iter()).map(|(acc,i)| {
                    acc.par_iter().zip(i.par_iter()).map(|(acc, i)| {
                        acc.par_iter().zip(i.par_iter()).map(|(&acc, &i)| acc + i).collect::<Vec<U>>().try_into()
                    }).collect::<Result<Vec<Arr<U, W>>, SizeMismatchError>>()?.try_into()
                }).collect::<Result<Vec<Image<U,H,W>>,SizeMismatchError>>()?.try_into()
            }))
        })?)
    }
    fn backward_weight_gradient_convolution<'a>(&self,loss: ImagesView<'a,U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                            input: ImagesView<'a,U, C, H, W>)
        -> Result<Arr4<U, K, C, FH, FW>, TrainingError> {
        Ok(loss.par_iter().map(|l| {
            input.par_iter().map(|i| {
                l.par_iter().zip(expand_image::<U, H, W, FH, FW, PAD, S>(i)?.par_iter()).map(|(l,i)| {
                    l.par_iter().zip(i).map(|(&l,i)| {
                        i.iter().map(|r| {
                            r.iter().map(|&p| {
                                l * p
                            }).collect::<Vec<U>>()
                        }).collect::<Vec<Vec<U>>>()
                    }).collect::<Vec<Vec<Vec<U>>>>()
                }).collect::<Vec<Vec<Vec<Vec<U>>>>>().into_par_iter().fold(|| Ok(Arr2::new()), | acc, i | {
                    acc.and_then(|acc| {
                        let i = i.into_par_iter().fold(|| Ok(Arr2::<U,FH,FW>::new()),| acc, i | {
                            acc.and_then(|acc| acc.iter().zip(i.iter()).map(|(acc,i)| {
                                acc.iter().zip(i.iter()).map(|(&acc,&p)| {
                                    acc + p
                                }).collect::<Vec<U>>().try_into()
                            }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into())
                        }).reduce(|| Ok(Arr2::new()), | acc, i | {
                            acc.and_then(|acc| i.and_then(|i| acc.iter().zip(i.iter()).map(|(acc,i)| {
                                acc.iter().zip(i.iter()).map(|(&acc,&p)| {
                                    acc + p
                                }).collect::<Vec<U>>().try_into()
                            }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into()))
                        })?;

                        acc.iter().zip(i.iter()).map(|(acc,i)| {
                            acc.iter().zip(i.iter()).map(|(&acc,&p)| {
                                acc + p
                            }).collect::<Vec<U>>().try_into()
                        }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into()
                    })
                }).reduce(|| Ok(Arr2::new()), | acc, i | {
                    acc.and_then(|acc| i.and_then(|i| Ok(acc.iter().zip(i.iter()).map(|(acc,i)| {
                        acc.iter().zip(i.iter()).map(|(&acc,&p)| {
                            acc + p
                        }).collect::<Vec<U>>().try_into()
                    }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into()?)))
                })
            }).collect::<Result<Vec<Arr2<U,FH,FW>>,SizeMismatchError>>()?.try_into()
        }).collect::<Result<Vec<Arr3<U,C,FH,FW>>,SizeMismatchError>>()?.try_into()?)
    }
    fn batch_forward_convolution<'a>(&self, input: SerializedVecView<'a,U,Images<U, C, H, W>>, kernel: &Arr4<U,K,C,FH,FW>)
        -> Result<SerializedVec<U, Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>, EvaluateError> {
        Ok(input.par_iter().map(|i| {
            self.forward_convolution(i,kernel)
        }).collect::<Result<Vec<Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,EvaluateError>>()?.into())
    }
    fn batch_backward_convolution<'a>(&self,
                                  loss: SerializedVecView<'a,U, Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                  kernel: &Arr4<U,K,C,FH,FW>)
        -> Result<SerializedVec<U, Images<U, C, H, W>>, TrainingError> {
        Ok(loss.par_iter().map(|l| {
            self.backward_convolution(l,kernel)
        }).collect::<Result<Vec<Images<U,C,H,W>>,TrainingError>>()?.into())
    }
    fn batch_backward_weight_gradient_convolution<'a>(&self,
                                                  loss: SerializedVecView<'a,U, Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                                  input: SerializedVecView<'a,U, Images<U, C, H, W>>)
        -> Result<Arr4<U,K,C,FH,FW>, TrainingError> {
        Ok(loss.par_iter().zip(input.par_iter()).map(|(l,i)| {
            self.backward_weight_gradient_convolution(l,i)
        }).collect::<Result<Vec<Arr4<U,K,C,FH,FW>>,TrainingError>>()?.par_iter().fold(|| Ok(Arr4::new()), | acc, k | {
            acc.and_then(|acc| acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                    acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                        acc.par_iter().zip(k.par_iter()).map(|(&acc,&k)| {
                            acc + k
                        }).collect::<Vec<U>>().try_into()
                    }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into()
                }).collect::<Result<Vec<Arr2<U,FH,FW>>,SizeMismatchError>>()?.try_into()
            }).collect::<Result<Vec<Arr3<U,C,FH,FW>>,SizeMismatchError>>()?.try_into())
        }).reduce(|| Ok(Arr4::new()), | acc, k | {
            acc.and_then(|acc| k.and_then(|k| acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                    acc.par_iter().zip(k.par_iter()).map(|(acc,k)| {
                        acc.par_iter().zip(k.par_iter()).map(|(&acc,&k)| {
                            acc + k
                        }).collect::<Vec<U>>().try_into()
                    }).collect::<Result<Vec<Arr<U,FW>>,SizeMismatchError>>()?.try_into()
                }).collect::<Result<Vec<Arr2<U,FH,FW>>,SizeMismatchError>>()?.try_into()
            }).collect::<Result<Vec<Arr3<U,C,FH,FW>>,SizeMismatchError>>()?.try_into()))
        })?)
    }
}