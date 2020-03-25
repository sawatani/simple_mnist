{-# LANGUAGE AllowAmbiguousTypes   #-}
{-# LANGUAGE ConstraintKinds       #-}
{-# LANGUAGE FlexibleContexts      #-}
{-# LANGUAGE FlexibleInstances     #-}
{-# LANGUAGE InstanceSigs          #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE QuasiQuotes           #-}
{-# LANGUAGE TypeFamilies          #-}

module Layers
  ( ForwardLayer(..)
  , OutputLayer(..)
  , ForwardNN(..)
  , BackwardLayer(..)
  , BackputLayer(..)
  , BackwardNN(..)
  , AffineForward(..)
  , SigmoidForward(..)
  , ReLUForward(..)
  , JoinedForwardLayer(..)
  , AffineBackward(..)
  , SigmoidBackward(..)
  , ReLUBackward(..)
  , JoinedBackwardLayer(..)
  , TrainBatch(..)
  , NElement(..)
  , (<~)
  , (~>)
  , learnForward
  , learnBackward
  , learn
  , learnAll
  , predict
  , evaluate
  ) where

import           Control.Arrow
import           Control.Lens            hiding ((<~))
import           Data.String.Interpolate as S (i)
import           Debug.Trace
import           Numeric
import           Numeric.LinearAlgebra
import           Synapse

type NElement a = (Ord a, Floating a, Numeric a, Num (Vector a), Show a)

type Weight a = Matrix a

type Bias a = Vector a

type SignalX a = Matrix a

type SignalY a = Matrix a

type Diff a = Matrix a

type TeacherBatch a = Matrix a

type InputBatch a = Matrix a

newtype TrainBatch a =
  TrainBatch (TeacherBatch a, InputBatch a)
  deriving (Show)

{-
Forward Layers
-}
class ForwardLayer (a :: * -> *) e where
  type Backward a :: * -> *
  forward :: NElement e => a e -> SignalX e -> (Backward a e, SignalY e)

class OutputLayer (o :: * -> *) e where
  type Backput o :: * -> *
  output :: NElement e => o e -> TeacherBatch e -> SignalY e -> (Backput o e, e)

infixl 4 ~>

(~>) ::
     (ForwardLayer a e, ForwardLayer b e, ForwardLayer c e) => a e -> b e -> c e
a ~> (JoinedForwardLayer x y) = (a ~> x) ~> y
a ~> b = JoinedForwardLayer a b

data AffineForward e =
  AffineForward (Weight e) (Bias e)
  deriving (Show)

data SigmoidForward e =
  SigmoidForward
  deriving (Show, Eq)

data ReLUForward e =
  ReLUForward
  deriving (Show, Eq)

data JoinedForwardLayer a b e =
  JoinedForwardLayer (a e) (b e)

data SoftmaxWithCrossForward e =
  SoftmaxWithCrossForward
  deriving (Show)

data ForwardNN a b e =
  ForwardNN (a e) (b e)
  deriving (Show)

instance ForwardLayer AffineForward e where
  type Backward AffineForward = AffineBackward
  forward (AffineForward w b) x = (AffineBackward w b x, affinem w b x)

instance ForwardLayer SigmoidForward e where
  type Backward SigmoidForward = SigmoidBackward
  forward SigmoidForward x = (SigmoidBackward y, y)
    where
      y = sigmoidm x

instance ForwardLayer ReLUForward e where
  type Backward ReLUForward = ReLUBackward
  forward ReLUForward x = (ReLUBackward x, relum x)

instance (ForwardLayer a e, ForwardLayer b e) =>
         ForwardLayer (JoinedForwardLayer a b) e where
  type Backward (JoinedForwardLayer a b) = JoinedBackwardLayer (Backward a) (Backward b)
  forward (JoinedForwardLayer a b) x0 = (a' <~ b', x2)
    where
      (a', x1) = forward a x0
      (b', x2) = forward b x1

instance (Numeric e, Eq e) => Eq (AffineForward e) where
  AffineForward w b == AffineForward w' b' = w == w' && b == b'

instance (Eq (a e), Eq (b e)) => Eq (JoinedForwardLayer a b e) where
  JoinedForwardLayer a b == JoinedForwardLayer a' b' = a == a' && b == b'

instance OutputLayer SoftmaxWithCrossForward e where
  type Backput SoftmaxWithCrossForward = SoftmaxWithCrossBackward
  output SoftmaxWithCrossForward t y = (SoftmaxWithCrossBackward t y', loss)
    where
      (y', loss) = softmaxWithCross t y

{-
Backward Layers
-}
class BackwardLayer (b :: * -> *) e where
  type Forward b :: * -> *
  backward :: NElement e => e -> b e -> Diff e -> (Forward b e, Diff e)

class BackputLayer (b :: * -> *) e where
  type Output b e :: *
  backput :: NElement e => b e -> (Output b e, Diff e)

infixr 4 <~

(<~) ::
     (BackwardLayer a e, BackwardLayer b e, BackwardLayer c e)
  => a e
  -> b e
  -> c e
(JoinedBackwardLayer x y) <~ b = x <~ (y <~ b)
a <~ b = JoinedBackwardLayer a b

data AffineBackward e =
  AffineBackward (Weight e) (Bias e) (SignalX e)
  deriving (Show)

data SigmoidBackward e =
  SigmoidBackward (SignalY e)
  deriving (Show)

data ReLUBackward e =
  ReLUBackward (SignalX e)
  deriving (Show)

data JoinedBackwardLayer a b e =
  JoinedBackwardLayer (a e) (b e)

data SoftmaxWithCrossBackward e =
  SoftmaxWithCrossBackward (TeacherBatch e) (SignalY e)

data BackwardNN a b e =
  BackwardNN (a e) (b e)

instance BackwardLayer AffineBackward e where
  type Forward AffineBackward = AffineForward
  backward rate (AffineBackward w b x) d =
    (AffineForward (w - scale rate w') (b - scale rate b'), x')
    where
      (x', w', b') = affinemBackward w x d

instance BackwardLayer SigmoidBackward e where
  type Forward SigmoidBackward = SigmoidForward
  backward _ (SigmoidBackward y) d = (SigmoidForward, sigmoidBackward y d)

instance BackwardLayer ReLUBackward e where
  type Forward ReLUBackward = ReLUForward
  backward _ (ReLUBackward x) d = (ReLUForward, relumBackward x d)

instance (BackwardLayer a e, BackwardLayer b e) =>
         BackwardLayer (JoinedBackwardLayer a b) e where
  type Forward (JoinedBackwardLayer a b) = JoinedForwardLayer (Forward a) (Forward b)
  backward r (JoinedBackwardLayer a b) d0 = (a' ~> b', d2)
    where
      (b', d1) = backward r b d0
      (a', d2) = backward r a d1

instance (Numeric e, Eq e) => Eq (AffineBackward e) where
  AffineBackward w b d == AffineBackward w' b' d' =
    w == w' && b == b' && d == d'

instance (Numeric e, Eq e) => Eq (SigmoidBackward e) where
  SigmoidBackward y == SigmoidBackward y' = y == y'

instance (Numeric e, Eq e) => Eq (ReLUBackward e) where
  ReLUBackward x == ReLUBackward x' = x == x'

instance (Eq (a e), Eq (b e)) => Eq (JoinedBackwardLayer a b e) where
  JoinedBackwardLayer a b == JoinedBackwardLayer a' b' = a == a' && b == b'

instance BackputLayer SoftmaxWithCrossBackward e where
  type Output SoftmaxWithCrossBackward e = SoftmaxWithCrossForward e
  backput (SoftmaxWithCrossBackward t y) =
    (SoftmaxWithCrossForward, softmaxWithCrossBackward t y)

{-
Learning
-}
learnForward ::
     ( NElement e
     , ForwardLayer a e
     , OutputLayer b e
     , BackwardLayer a' e
     , BackputLayer b' e
     )
  => ForwardNN a b e
  -> TrainBatch e
  -> (BackwardNN a' b' e, e)
learnForward (ForwardNN layers loss) (TrainBatch (t, x)) =
  result `seq` (BackwardNN layers' loss', result)
  where
    (layers', y) = forward layers x
    (loss', result) = output loss t y

learnBackward ::
     ( NElement e
     , ForwardLayer a e
     , OutputLayer b e
     , BackwardLayer a' e
     , BackputLayer b' e
     )
  => e
  -> BackwardNN a' b' e
  -> ForwardNN a b e
learnBackward rate (BackwardNN layers loss) = ForwardNN layers' loss'
  where
    (loss', d) = backput loss
    (layers', _) = backward rate layers d

learn ::
     (NElement e, ForwardLayer a e, OutputLayer b e)
  => e
  -> ForwardNN a b e
  -> TrainBatch e
  -> (ForwardNN a b e, e)
learn rate a = first (learnBackward rate) . learnForward a

learnAll ::
     (NElement e, ForwardLayer a e, OutputLayer b e)
  => e
  -> ForwardNN a b e
  -> [TrainBatch e]
  -> (ForwardNN a b e, [e])
learnAll rate origin batches = ls `seq` (nn, ls)
  where
    (nn, ls) = foldr f (origin, []) batches
    f batch (a, ls) = ls `seq` second (~: ls) $ learn rate a batch
    x ~: xs = trace [i|[#{length xs + 1}/#{n}] #{show x}|] (x : xs)
    n = length batches

predict :: (NElement e, ForwardLayer a e) => a e -> Vector e -> Int
predict layers = maxIndex . flatten . snd . forward layers . asRow

evaluate :: (NElement e, ForwardLayer a e) => a e -> [(Int, Vector e)] -> Double
evaluate layers samples = fromIntegral nOk / fromIntegral (length samples)
  where
    nOk = length $ filter (uncurry (==)) results
    results = map (second $ predict layers) samples
