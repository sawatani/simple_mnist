module Layers
    (
      ForwardLayer(..)
    , OutputLayer(..)
    , ForwardNN(..)
    , BackwardLayer(..)
    , BackputLayer(..)
    , BackwardNN(..)
    , TrainBatch(..)
    , NElement(..)
    , (<~)
    , (~>)
    , forward
    , output
    , backward
    , backput
    , learnForward
    , learnBackward
    , learn
    , learnAll
    , predict
    , evaluate
    ) where

import Synapse
import qualified Codec.Compression.GZip      as GZip
import           Control.Arrow
import           Control.DeepSeq
import           Control.Lens hiding ((<~))
import           Control.Monad               (join)
import           Control.Monad.ST
import           Control.Monad.State
import           Control.Monad.Trans.Class
import           Data.Binary                 as B
import           Data.Binary.Get.Internal    as BGI
import qualified Data.ByteString             as BSS
import qualified Data.ByteString.Lazy        as BS
import qualified Data.ByteString.Lazy.UTF8   as UTF8
import qualified Data.Conduit                as CD
-- import qualified Data.CSV.Conduit            as CSV
import           Data.Int
import           Data.List
import           Data.List.Split
import qualified Data.Map                    as Map
import           Data.Time.Clock
import           Data.Time.Format
import           Data.Time.LocalTime
import qualified Data.Vector                 as V
import qualified Data.Vector.Generic         as VG
import qualified Data.Vector.Generic.Mutable as VM
import           Debug.Trace
import           Network.HTTP.Client
import           Network.HTTP.Types.Status   (statusCode)
import           Numeric
import           Numeric.LinearAlgebra
import           Numeric.LinearAlgebra.Data
import           Prelude                     hiding ((<>))
import           System.Directory
import           System.FilePath.Posix
import           System.IO
import           System.Random.MWC

type Weight a = Matrix a
type Bias a = Vector a
type SignalX a = Matrix a
type SignalY a = Matrix a
type Diff a = Matrix a
type TeacherBatch a = Matrix a
type InputBatch a = Matrix a
newtype TrainBatch a = TrainBatch (TeacherBatch a, InputBatch a) deriving (Show)

data ForwardLayer a =
    AffineForward (Weight a) (Bias a)
  | SigmoidForward
  | ReLUForward
  | JoinedForwardLayer (ForwardLayer a) (ForwardLayer a)
  deriving (Show)

infixl 4 ~>
(~>) :: ForwardLayer a -> ForwardLayer a -> ForwardLayer a
a ~> (JoinedForwardLayer x y) = (a ~> x) ~> y
a ~> b = JoinedForwardLayer a b

data OutputLayer a = SoftmaxWithCrossForward
  deriving (Show)

data ForwardNN a = ForwardNN (ForwardLayer a) (OutputLayer a)
  deriving (Show)

data BackwardLayer a =
    AffineBackward (Weight a) (Bias a) (SignalX a)
  | SigmoidBackward (SignalY a)
  | ReLUBackward (SignalX a)
  | JoinedBackwardLayer (BackwardLayer a) (BackwardLayer a)
  deriving (Show)

infixr 4 <~
(<~) :: BackwardLayer a -> BackwardLayer a -> BackwardLayer a
(JoinedBackwardLayer x y) <~ b = x <~ (y <~ b)
a <~ b = JoinedBackwardLayer a b

data BackputLayer a = SoftmaxWithCrossBackward (TeacherBatch a) (SignalY a)

data BackwardNN a = BackwardNN (BackwardLayer a) (BackputLayer a)

type NElement a = (Ord a, Floating a, Numeric a, Num (Vector a), Show a)

forward :: NElement a => ForwardLayer a -> SignalX a -> (BackwardLayer a, SignalY a)
forward (AffineForward w b) x = (AffineBackward w b x, affinem w b x)
forward SigmoidForward x = let y = sigmoidm x in (SigmoidBackward y, y)
forward ReLUForward x = (ReLUBackward x, relum x)
forward (JoinedForwardLayer a b) x0 = (a' <~ b', x2)
    where
        (a', x1) = forward a x0
        (b', x2) = forward b x1

backward :: NElement a => a -> BackwardLayer a -> Diff a -> (ForwardLayer a, Diff a)
backward rate (AffineBackward w b x) d =  (AffineForward (w - scale rate w') (b - scale rate b'), x')
    where (x', w', b') = affinemBackward w x d
backward _ (SigmoidBackward y) d = (SigmoidForward, sigmoidBackward y d)
backward _ (ReLUBackward x) d = (ReLUForward, relumBackward x d)
backward r (JoinedBackwardLayer a b) d0 = (a' ~> b', d2)
    where
        (b', d1) = backward r b d0
        (a', d2) = backward r a d1

output :: NElement a => OutputLayer a -> TeacherBatch a -> SignalY a -> (BackputLayer a, a)
output SoftmaxWithCrossForward t y = (SoftmaxWithCrossBackward t y', loss)
    where
        (y', loss) = softmaxWithCross t y

backput :: NElement a => BackputLayer a -> (OutputLayer a, Diff a)
backput (SoftmaxWithCrossBackward t y) = (SoftmaxWithCrossForward, softmaxWithCrossBackward t y)


learnForward :: NElement a => ForwardNN a -> TrainBatch a -> (BackwardNN a, a)
learnForward (ForwardNN layers loss) (TrainBatch (t, x)) = result `seq` (BackwardNN layers' loss', result)
    where
        (layers', y) = forward layers x
        (loss', result) = output loss t y

learnBackward :: NElement a => a -> BackwardNN a -> ForwardNN a
learnBackward  rate (BackwardNN layers loss) = ForwardNN layers' loss'
    where
        (loss', d) = backput loss
        (layers', _) = backward rate layers d

learn :: NElement a => a -> ForwardNN a -> TrainBatch a -> (ForwardNN a, a)
learn rate a = first (learnBackward rate) . learnForward a

learnAll :: NElement a => a -> ForwardNN a -> [TrainBatch a] -> (ForwardNN a, [a])
learnAll rate origin = foldr f (origin, [])
    where f batch (a, results) = trace ("=========\nLearning iterate: " ++ show (length results + 1)) $ a `seq` second (:results) $ learn rate a batch

predict :: NElement a => ForwardLayer a -> Vector a -> Int
-- predict layers = maxIndex . flatten . snd . (forward layers) . asRow
predict layers sample = maxIndex $ trace ("Evaluated result: " ++ show result) result
    where
        m1 = asRow sample
        (_, m2) = forward layers m1
        result = flatten m2

evaluate :: NElement a => ForwardLayer a -> [(Int, Vector a)] -> Double
evaluate layers samples = fromIntegral nOk / fromIntegral (length samples)
    where
        nOk = length $ filter (uncurry (==)) results
        results = map (second $ predict layers) samples
