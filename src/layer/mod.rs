use std::fmt::Debug;
use std::str::FromStr;

use nncombinator::arr::{Arr4, VecArr};
use nncombinator::{Cons, Stack};
use nncombinator::device::{Device, DeviceCpu};
use nncombinator::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use nncombinator::layer::{AskDiffInput, Backward, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain};
use nncombinator::lossfunction::LossFunction;
use nncombinator::ope::UnitValue;
use nncombinator::optimizer::Optimizer;
use nncombinator::persistence::*;
use crate::collection::{Images};
use crate::device::DeviceConvolution;
use crate::{Assert, assert_convolution, IsTrue};

pub trait ConvolutionLayerInstantiation<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    /// * `ui` - Callback to generate weight of unit
    ///
    fn new<UI: FnMut() -> U>(parent:P,device:&D,ui:UI) -> Result<ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>,LayerInstantiationError>;
}
///  Convolution Layer Implementation
pub struct ConvolutionLayer<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          F: Send + Sync + 'static,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    parent:P,
    device:D,
    kernel:F
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
        const FH: usize,const FW: usize,const PAD:usize,const S:usize
    > ConvolutionLayerInstantiation<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn new<UI: FnMut() -> U>(parent:P,device:&DeviceCpu<U>,ui:UI)
        -> Result<ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>,LayerInstantiationError> {
        let mut kernel = Arr4::new();

        let mut ui = ui;

        for mut k in kernel.iter_mut() {
            for mut c in k.iter_mut() {
                for mut h in c.iter_mut() {
                    for w in h.iter_mut() {
                        *w = ui();
                    }
                }
            }
        }

        Ok(ConvolutionLayer {
            parent: parent,
            device: device.clone(),
            kernel: kernel
        })
    }
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
        const FH: usize,const FW: usize,const PAD:usize,const S:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + Send + UnitValue<U>+ FromStr,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for mut k in self.kernel.iter_mut() {
            for mut c in k.iter_mut() {
                for mut h in c.iter_mut() {
                    for w in h.iter_mut() {
                        *w = persistence.read()?;
                    }
                }
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        persistence.write(UnitOrMarker::LayerStart);

        for k in self.kernel.iter() {
            persistence.write(UnitOrMarker::UnitsStart);

            for c in k.iter() {
                for h in c.iter() {
                    for w in h.iter() {
                        persistence.write(UnitOrMarker::Unit(*w));
                    }
                }
            }
        }

        Ok(())
    }
}
impl<T,U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Persistence<U,T,Linear>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        for mut k in self.kernel.iter_mut() {
            for mut c in k.iter_mut() {
                for mut h in c.iter_mut() {
                    for w in h.iter_mut() {
                        *w = persistence.read()?;
                    }
                }
            }
        }

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        for k in self.kernel.iter() {
            for c in k.iter() {
                for h in c.iter() {
                    for w in h.iter() {
                        persistence.write(*w)?;
                    }
                }
            }
        }

        Ok(())
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
        Forward<Images<U,C,H,W>,Result<Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,EvaluateError>>
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn forward(&self,input:&Images<U,C,H,W>)
        -> Result<Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,EvaluateError> {
        self.device.forward_convolution(input,&self.kernel)
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> ForwardAll for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type Input = I;
    type Output = Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> PreTrain<U> for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type OutStack = Cons<<P as PreTrain<U>>::OutStack,Self::Output>;

    fn pre_train(&self, input: Self::Input) -> Result<Self::OutStack, EvaluateError> {
        let s = self.parent.pre_train(input)?;

        let o = s.map(|input| self.forward(input))?;

        Ok(s.push(o))
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
        Backward<U,&Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,Result<Images<U,C,H,W>,TrainingError>>
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn backward(&mut self, loss: &Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>)
        -> Result<Images<U,C,H,W>,TrainingError> {
        self.device.backward_convolution(loss,&self.kernel)
    }
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BackwardAll<U>
        for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type LossInput = Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L)
        -> Result<(), TrainingError> {

        let (s,_) = stack.pop();

        let loss = input;

        s.map(|i| {
            self.device.backward_weight_gradient_convolution(&loss,i).map(|g| {
                for (mut k,g) in self.kernel.iter_mut().zip(g.iter()) {
                    for (mut c,g) in k.iter_mut().zip(g.iter()) {
                        for (mut h,g) in c.iter_mut().zip(g.iter()) {
                            for (w,&g) in h.iter_mut().zip(g.iter()) {
                                optimizer.update(g, w);
                            }
                        }
                    }
                }
            })
        })?;

        let loss = self.device.backward_convolution(&loss,&self.kernel)?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss.into(), s, optimizer, lossf)
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> AskDiffInput<U> for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Self: PreTrain<U>,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Loss<U>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchForwardBase
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + 'static,
          D: Device<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: ForwardAll {
    type BatchInput = VecArr<U,I>;
    type BatchOutput = VecArr<U,Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>;
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchForward
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + BatchForward + 'static,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_convolution(&input,&self.kernel)?)
    }
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchPreTrainBase<U>
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + 'static,
          D: Device<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,P,D,I,F,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchPreTrain<U>
    for ConvolutionLayer<U,P,D,I,F,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + 'static,
          D: Device<U> + DeviceConvolution<U,F,C,K,H,W,FH,FW,PAD,S> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          F: Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: PreTrain<U> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_convolution(input,&self.kernel))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchBackward<U>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Images<U,C,H,W>>> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type BatchLossInput = VecArr<U,Images<U,K,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s, _) = stack.pop();

        let loss = input;

        {
            s.map(|o| {
                self.device.batch_backward_weight_gradient_convolution(&loss,o).map(|g| {
                    for (mut k,g) in self.kernel.iter_mut().zip(g.iter()) {
                        for (mut c,g) in k.iter_mut().zip(g.iter()) {
                            for (mut h,g) in c.iter_mut().zip(g.iter()) {
                                for (w,&g) in h.iter_mut().zip(g.iter()) {
                                    optimizer.update(g, w);
                                }
                            }
                        }
                    }
                })
            })?;
        }

        let loss = self.device.batch_backward_convolution(&loss,&self.kernel)?;

        let (s,loss) = self.parent.batch_loss(loss,lossf,s)?;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const C:usize,const K:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchLoss<U>
    for ConvolutionLayer<U,P,DeviceCpu<U>,I,Arr4<U,K,C,H,W>,C,K,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=VecArr<U,I>,BatchOutput=VecArr<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=VecArr<U,Images<U,C,H,W>>> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          VecArr<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
}