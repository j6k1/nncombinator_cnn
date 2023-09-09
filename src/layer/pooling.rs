use std::fmt::Debug;
use std::str::FromStr;

use nncombinator::arr::{SerializedVec};
use nncombinator::{Cons, Stack};
use nncombinator::device::{Device, DeviceCpu};
use nncombinator::error::{ConfigReadError, EvaluateError, LayerInstantiationError, PersistenceError, TrainingError};
use nncombinator::layer::{AskDiffInput, BackwardAll, BatchBackward, BatchForward, BatchForwardBase, BatchLoss, BatchPreTrain, BatchPreTrainBase, Forward, ForwardAll, Loss, PreTrain};
use nncombinator::lossfunction::LossFunction;
use nncombinator::ope::UnitValue;
use nncombinator::optimizer::Optimizer;
use nncombinator::persistence::*;
use crate::collection::{Images};
use crate::{Assert, assert_convolution, IsTrue};
use crate::device::pooling::DeviceMaxPooling2D;

pub trait MaxPooling2DLayerInstantiation<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    /// Create and return an instance with the specified scale, bias, and momentum.
    /// # Arguments
    /// * `parent` - upper layer
    /// * `device` - Device object used for neural network computation
    ///
    fn new(parent:P,device:&D) -> Result<MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>,LayerInstantiationError>;
}
///  Convolution Layer Implementation
pub struct MaxPooling2DLayer<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    parent:P,
    device:D
}
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> MaxPooling2DLayerInstantiation<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn new(parent:P,device:&DeviceCpu<U>)
        -> Result<MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>,LayerInstantiationError> {

        Ok(MaxPooling2DLayer {
            parent: parent,
            device: device.clone()
        })
    }
}
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Persistence<U,TextFilePersistence<U>,Specialized>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
              BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + Persistence<U,TextFilePersistence<U>,Specialized>,
          U: Default + Clone + Copy + Send + UnitValue<U>+ FromStr,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          ConfigReadError: From<<U as FromStr>::Err> {
    fn load(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut TextFilePersistence<U>) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        Ok(())
    }
}
impl<T,U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Persistence<U,T,Linear>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where T: LinearPersistence<U>,
          P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
          BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + Persistence<U,T,Linear>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn load(&mut self, persistence: &mut T) -> Result<(),ConfigReadError> {
        self.parent.load(persistence)?;

        Ok(())
    }

    fn save(&mut self, persistence: &mut T) -> Result<(), PersistenceError> {
        self.parent.save(persistence)?;

        Ok(())
    }
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Forward<Images<U,C,H,W>,
    Result<Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,EvaluateError>>
    for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue {
    fn forward(&self,input:&Images<U,C,H,W>)
               -> Result<Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>,EvaluateError> {
        self.device.forward_maxpooling_2d(input)
    }
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> ForwardAll for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type Input = I;
    type Output = Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>;
    fn forward_all(&self, input: Self::Input) -> Result<Self::Output, EvaluateError> {
        self.forward(&self.parent.forward_all(input)?)
    }
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> PreTrain<U> for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U> + DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> + 'static,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
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
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BackwardAll<U>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type LossInput = Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>;

    fn backward_all<OP: Optimizer<U>,L: LossFunction<U>>(&mut self, input: Self::LossInput, stack:Self::OutStack, optimizer: &mut OP, lossf:&L)
                                                         -> Result<(), TrainingError> {

        let (s,o) = stack.pop();

        let loss = input;

        let loss = s.map(|i| {
            self.device.backward_maxpooling_2d(&loss,i,&o)
        })?;

        let (s,loss) = self.parent.loss(loss,lossf,s)?;

        self.parent.backward_all(loss.into(), s, optimizer, lossf)
    }
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> AskDiffInput<U> for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: PreTrain<U,OutStack=<<Self as PreTrain<U>>::OutStack as Stack>::Remaining> +
             ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + Loss<U> + AskDiffInput<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          D: Device<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Self: PreTrain<U>,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type DiffInput = P::DiffInput;

    fn ask_diff_input(&self, stack: &Self::OutStack) -> Self::DiffInput {
        stack.map_remaining(|s| self.parent.ask_diff_input(s))
    }
}
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> Loss<U>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchForwardBase
    for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + 'static,
          D: Device<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: ForwardAll {
    type BatchInput = SerializedVec<U,I>;
    type BatchOutput = SerializedVec<U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>;
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchForward
    for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + BatchForward + 'static,
          D: Device<U> + DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    fn batch_forward(&self, input: Self::BatchInput) -> Result<Self::BatchOutput, TrainingError> {
        let input = self.parent.batch_forward(input)?;

        Ok(self.device.batch_forward_maxpooling_2d(&input)?)
    }
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchPreTrainBase<U>
    for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + 'static,
          D: Device<U>,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: PreTrain<U> {
    type BatchOutStack = Cons<<P as BatchPreTrainBase<U>>::BatchOutStack,Self::BatchOutput>;
}
impl<U,P,D,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchPreTrain<U>
    for MaxPooling2DLayer<U,P,D,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchPreTrain<U> + 'static,
          D: Device<U> + DeviceMaxPooling2D<U,C,H,W,FH,FW,PAD,S> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]:,
          Self: PreTrain<U> {
    fn batch_pre_train(&self, input: Self::BatchInput) -> Result<Self::BatchOutStack, TrainingError> {
        let r = self.parent.batch_pre_train(input)?;

        let u = r.map(|input| self.device.batch_forward_maxpooling_2d(input))?;

        Ok(Cons(r,u))
    }
}
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchBackward<U>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Images<U,C,H,W>>> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
    type BatchLossInput = SerializedVec<U,Images<U,C,{ ( H + 2 * PAD - FH ) / S + 1 }, { ( W + 2 * PAD - FW ) / S + 1 }>>;

    fn batch_backward<OP: Optimizer<U>, L: LossFunction<U>>(&mut self, input: Self::BatchLossInput, stack: Self::BatchOutStack, optimizer: &mut OP, lossf: &L) -> Result<(), TrainingError> {
        let (s, o) = stack.pop();

        let loss = input;

        let loss = s.map(|i| {
            self.device.batch_backward_maxpooling_2d(&loss,i,&o)
        })?;

        let (s,loss) = self.parent.batch_loss(loss,lossf,s)?;

        self.parent.batch_backward(loss, s, optimizer, lossf)
    }
}
impl<U,P,I,const C:usize,const H:usize,const W:usize,
    const FH: usize,const FW: usize,const PAD:usize,const S:usize> BatchLoss<U>
    for MaxPooling2DLayer<U,P,DeviceCpu<U>,I,C,H,W,FH,FW,PAD,S>
    where P: ForwardAll<Input=I,Output=Images<U,C,H,W>> +
             BackwardAll<U,LossInput=Images<U,C,H,W>> + PreTrain<U> + Loss<U> +
             BatchForwardBase<BatchInput=SerializedVec<U,I>,BatchOutput=SerializedVec<U,Images<U,C,H,W>>> + BatchForward +
             BatchPreTrainBase<U> + BatchBackward<U> + BatchLoss<U,BatchLossInput=SerializedVec<U,Images<U,C,H,W>>> + 'static,
          U: Default + Clone + Copy + Send + UnitValue<U>,
          I: Debug + Send + Sync + 'static,
          SerializedVec<U,I>: Debug + Send + Sync + 'static,
          Assert<{ assert_convolution::<H,W,FH,FW,PAD,S>() }>: IsTrue,
          [(); (H + 2 * PAD - FH) / S + 1]:,
          [(); (W + 2 * PAD - FW) / S + 1]: {
}