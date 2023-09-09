use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use nncombinator::arr::{Arr,SerializedVec};
use nncombinator::device::DeviceCpu;
use nncombinator::error::{EvaluateError, SizeMismatchError, TrainingError};
use nncombinator::ope::UnitValue;
use crate::{Assert, assert_convolution, IsTrue};
use crate::collection::{Image, Images};

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
    fn forward_maxpooling_2d(&self, input:&Images<U,C,H,W>)
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
    fn backward_maxpooling_2d(&self, loss:&Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                              input:&Images<U,C,H,W>,
                              output:&Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>)
                              -> Result<Images<U,C,H,W>, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    ///* [`EvaluateError`]
    fn batch_forward_maxpooling_2d(&self, input:&SerializedVec<U,Images<U,C,H,W>>)
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
    fn batch_backward_maxpooling_2d(&self,loss:&SerializedVec<U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                    input:&SerializedVec<U,Images<U,C,H,W>>,
                                    output:&SerializedVec<U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>)
                                    -> Result<SerializedVec<U,Images<U,C,H,W>>, TrainingError>;
}
impl<U,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> for DeviceCpu<U>
    where U: UnitValue<U>,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn forward_maxpooling_2d(&self, input: &Images<U, C, H, W>)
        -> Result<Images<U,C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError> {
        Ok(input.par_iter().map(|i| {
            (0..(H + PAD * 2 - FH)).into_par_iter().step_by(S).map(|sy| {
                (0..(W + PAD * 2 - FW)).into_par_iter().step_by(S).map(|sx| {
                    i.iter().skip(sy - PAD.min(sy)).enumerate().map(|(dy,i)| {
                        (dy + PAD - PAD.min(sy),i)
                    }).take_while(|(dy,_)| {
                        *dy < FH && sy + *dy < H + PAD
                    }).fold(U::initial_max_value(),|acc,(_,r)| {
                        r.iter().skip(sx - PAD.min(sx)).enumerate().map(|(dx,i)| {
                            (dx + PAD - PAD.min(sx),i)
                        }).take_while(|(dx,_)| {
                            *dx < FW && sx + *dx < W + PAD
                        }).fold(acc,|acc,(_,p)| {
                            p.max(&acc)
                        })
                    })
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()
        }).collect::<Result<Vec<Image<U,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()?)
    }

    fn backward_maxpooling_2d(&self, loss: &Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                              input: &Images<U, C, H, W>,
                              _: &Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>) -> Result<Images<U, C, H, W>, TrainingError> {
        Ok(input.par_iter().zip(loss.par_iter()).map(|(i,l)| {
            let mut result = Image::<U,H,W>::new();

            let indexes = (0..(H + PAD * 2 - FH)).into_par_iter().step_by(S).map(|sy| {
                (0..(W + PAD * 2 - FW)).into_par_iter().step_by(S).map(|sx| {
                    i.iter().skip(sy - PAD.min(sy)).enumerate().map(|(dy,i)| {
                        (dy + PAD - PAD.min(sy),i)
                    }).take_while(|(dy,_)| {
                        *dy < FH && sy + *dy < H + PAD
                    }).fold(((0,0),U::initial_max_value()),|acc,(dy,r)| {
                        r.iter().skip(sx - PAD.min(sx)).enumerate().map(|(dx,i)| {
                           (dx + PAD - PAD.min(sx),i)
                        }).take_while(|(dx,_)| {
                            sx + dx < W + PAD
                        }).fold(acc,|((x,y),max),(dx,&p)| {
                            if acc.1.is_nan() || acc.1 < p {
                                ((dy,dx),p)
                            } else {
                                ((y,x),max)
                            }
                        })
                    }).0
                }).collect::<Vec<(usize,usize)>>()
            }).collect::<Vec<Vec<(usize,usize)>>>();

            for ((sy,r),d) in indexes.iter().enumerate().zip(l.iter()) {
                for ((sx,(dy,dx)),&d) in r.iter().enumerate().zip(d.iter()) {
                    result[(sy * S + dy - PAD, sx * S + dx - PAD)] += d;
                }
            }

            result
        }).collect::<Vec<Image<U,H,W>>>().try_into()?)
    }

    fn batch_forward_maxpooling_2d(&self, input: &SerializedVec<U, Images<U, C, H, W>>)
        -> Result<SerializedVec<U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>, EvaluateError> {
        Ok(input.par_iter().map(|i| {
            i.par_iter().map(|i| {
                (0..(H + PAD * 2 - FH)).into_par_iter().step_by(S).map(|sy| {
                    (0..(W + PAD * 2 - FW)).into_par_iter().step_by(S).map(|sx| {
                        i.iter().skip(sy - PAD.min(sy)).enumerate().map(|(dy,i)| {
                            (dy + PAD - PAD.min(sy),i)
                        }).take_while(|(dy,_)| {
                            *dy < FH && sy + *dy < H + PAD
                        }).fold(U::initial_max_value(),|acc,(_,r)| {
                            r.iter().skip(sx - PAD.min(sx)).enumerate().map(|(dx,i)| {
                                (dx + PAD - PAD.min(sx),i)
                            }).take_while(|(dx,_)| {
                                *dx < FW && sx + dx < W + PAD
                            }).fold(acc,|acc,(_,p)| {
                                p.max(&acc)
                            })
                        })
                    }).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()
            }).collect::<Result<Vec<Image<U, { (H + 2 * PAD - FH) / S + 1 }, { (W + 2 * PAD - FW) / S + 1 }>>, SizeMismatchError>>()?.try_into()
        }).collect::<Result<Vec<Images<U,C,{ (H + 2 * PAD - FH) / S + 1 }, { (W + 2 * PAD - FW) / S + 1 }>>,SizeMismatchError>>()?.into())
    }

    fn batch_backward_maxpooling_2d(&self, loss: &SerializedVec<U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,
                                    input: &SerializedVec<U, Images<U, C, H, W>>,
                                    _: &SerializedVec<U, Images<U, C, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>) -> Result<SerializedVec<U, Images<U, C, H, W>>, TrainingError> {
        Ok(input.par_iter().zip(loss.par_iter()).map(|(i,l)| {
            i.par_iter().zip(l.par_iter()).map(|(i,l)| {
                let mut result = Image::<U,H,W>::new();

                let indexes = (0..(H + PAD * 2 - FH)).into_par_iter().step_by(S).map(|sy| {
                    (0..(W + PAD * 2 - FW)).into_par_iter().step_by(S).map(|sx| {
                        i.iter().skip(sy - PAD.min(sy)).enumerate().map(|(dy,i)| {
                            (dy + PAD - PAD.min(sy),i)
                        }).take_while(|(dy,_)| {
                            *dy < FH && sy + *dy < H + PAD
                        }).fold(((0,0),U::initial_max_value()),|acc,(dy,r)| {
                            r.iter().skip(sx - PAD.min(sx)).enumerate().map(|(dx,i)| {
                                (dx + PAD - PAD.min(sx),i)
                            }).take_while(|(dx,_)| {
                                sx + dx < W + PAD
                            }).fold(acc,|((x,y),max),(dx,&p)| {
                                if acc.1.is_nan() || acc.1 < p {
                                    ((dy,dx),p)
                                } else {
                                    ((y,x),max)
                                }
                            })
                        }).0
                    }).collect::<Vec<(usize,usize)>>()
                }).collect::<Vec<Vec<(usize,usize)>>>();

                for ((sy,r),d) in indexes.iter().enumerate().zip(l.iter()) {
                    for ((sx,(dy,dx)),&d) in r.iter().enumerate().zip(d.iter()) {
                        result[(sy * S + dy - PAD, sx * S + dx - PAD)] += d;
                    }
                }

                result
            }).collect::<Vec<Image<U,H,W>>>().try_into()
        }).collect::<Result<Vec<Images<U,C,H,W>>,SizeMismatchError>>()?.into())
    }
}