use const_guards::guard;
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};
use nncombinator::arr::{Arr, Arr4};
use nncombinator::device::DeviceCpu;
use nncombinator::error::{EvaluateError, TrainingError};
use nncombinator::error::SizeMismatchError;
use nncombinator::ope::UnitValue;

use crate::collection::{Image, Images};
use crate::collection::VecImages;

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
    fn forward_convolution<const CI:usize>(&self, input:&Images<U,C,H,W>, kernel:&F)
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
    fn backward_convolution(&self, loss:&Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
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
    fn backward_weight_gradient_convolution(&self,
                                loss: &Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                input: &Images<U,C,H,W>)
        -> Result<Arr4<U,K,C,FH,FW>, TrainingError>;
    /// Forward propagation calculation in batch
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    ///* [`EvaluateError`]
    fn batch_forward_convolution<const CI:usize>(&self, input:&VecImages<U,C,H,W>, kernel:&F)
        -> Result<VecImages<U,K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError>;
    /// Error back propagation calculation in batch
    /// # Arguments
    /// * `input` - Input values from upper layers
    /// * `kernel` - filter weights
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_convolution(&self,loss:&VecImages<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                  kernel:&F)
       -> Result<VecImages<U,C,H,K>, TrainingError>;
    /// Calculate the gradient of the weights in batch
    /// # Arguments
    /// * `loss` - loss
    /// * `input` - Input values from upper layers
    ///
    /// # Errors
    ///
    /// This function may return the following errors
    /// * [`TrainingError`]
    fn batch_backward_weight_gradient_convolution(&self,
                                loss: &VecImages<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                input: &VecImages<U,C,H,W>)
        -> Result<Arr4<U,K,C,H,W>, TrainingError>;
}
#[guard({ H + 2 * PAD >= FH && (H + 2 * PAD - FH) % S == 0 && W + 2 * PAD >= FW && (W + 2 * PAD - FW) % S == 0 })]
impl<U,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> DeviceConvolution<U,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S> for DeviceCpu<U>
    where U: UnitValue<U> {
    fn forward_convolution<const CI: usize>(&self, input: &Images<U, C, H, W>, kernel: &Arr4<U,K,C,H,W>)
        -> Result<Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError> {
        Ok(kernel.par_iter().map(|k| {
            k.par_iter().zip(input.par_iter()).map(|(k,i)| {
                (0..(H + PAD * 2 - FH)).into_par_iter().step_by(S).map(|sy| {
                    (0..(W + PAD * 2 - FW)).into_par_iter().step_by(S).map(|sx| {
                        k.iter().enumerate()
                            .skip_while(|(oy,_)| sy + oy < PAD)
                            .take_while(|(oy,_)| sy + oy < H).zip(i.clone().skip(sy - PAD))
                            .map(|((_,k), i)| {
                                k.iter().enumerate()
                                    .skip_while(|(ox,_)| sx + ox < PAD)
                                    .take_while(|(ox,_)| sx + ox < W).zip(i.iter().clone().skip(sx - PAD))
                                    .map(|((_,&k),&i)| i * k)
                                    .fold(U::default(), |acc, p| acc + p)
                            }).fold(U::default(), |acc, p| acc + p)
                        }).collect::<Vec<U>>()
                }).collect::<Vec<Vec<U>>>()
            }).fold(|| Ok(Image::new()),|acc,img| {
                acc.and_then(|acc| acc.par_iter().zip(img.par_iter()).map(|(acc,row)| {
                    acc.par_iter().zip(row.par_iter()).map(|(&acc,&pixel)| acc + pixel).collect::<Vec<U>>().try_into()
                }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into())
            }).reduce(|| Ok(Image::new()),|acc,img| {
                acc.and_then(|acc| img.and_then(|img| {
                    acc.par_iter().zip(img.par_iter()).map(|(acc,row)| {
                        acc.par_iter().zip(row.par_iter()).map(|(&acc,&pixel)| acc + pixel).collect::<Vec<U>>().try_into()
                    }).collect::<Result<Vec<Arr<U,{ ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()
                }))
            })
        }).collect::<Result<Vec<Image<U,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>,SizeMismatchError>>()?.try_into()?)
    }

    fn backward_convolution(&self, loss: &Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, kernel: &Arr4<U,K,C,H,W>)
        -> Result<Images<U, C, H, W>, TrainingError> {
        Ok(kernel.par_iter().map(|k| {
            (0..H).into_par_iter().map(|sy| {
                (0..W).into_par_iter().map(|sx| {
                    loss.par_iter().zip(k.par_iter()).map(|(l, k)| {
                        let l = l[((sy + PAD) / S, (sx + PAD) / S)];

                        k.iter().skip((sy + PAD) % S).step_by(S).map(|k| {
                            k.iter().skip((sx + PAD) % S).step_by(S).map(|&w| l * w).fold(U::default(), |acc, l| {
                                acc + l
                            })
                        }).fold(U::default(), |acc, l| acc + l)
                    }).reduce(|| U::default(), | acc, l | acc + l)
                }).collect::<Vec<U>>().try_into()
            }).collect::<Result<Vec<Arr<U,W>>,SizeMismatchError>>()?.try_into()
        }).collect::<Result<Vec<Image<U,H,W>>,SizeMismatchError>>()?.try_into()?)
    }
    fn backward_weight_gradient_convolution(&self,loss: &Images<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                            input: &Images<U, C, H, W>)
        -> Result<Arr4<U, K, C, FH, FW>, TrainingError> {
        todo!()
    }
    fn batch_forward_convolution<const CI: usize>(&self, input: &VecImages<U, C, H, W>, kernel: &Arr4<U,K,C,H,W>)
        -> Result<VecImages<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>, EvaluateError> {
        todo!()
    }
    fn batch_backward_convolution(&self,
                                  loss: &VecImages<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                  kernel: &Arr4<U,K,C,H,W>)
        -> Result<VecImages<U, C, H, K>, TrainingError> {
        todo!()
    }
    fn batch_backward_weight_gradient_convolution(&self,
                                                  loss: &VecImages<U, K, { ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,
                                                  input: &VecImages<U, C, H, W>)
        -> Result<Arr4<U, K, C, H, W>, TrainingError> {
        todo!()
    }
}