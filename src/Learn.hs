{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE QuasiQuotes      #-}

module Learn
  ( newManager
  , defaultManagerSettings
  , predict
  , evaluate
  , initNN
  , hotone
  , convertTrains
  , convertTests
  , trainingSimple
  , chunksOf
  ) where

import           Data.Bifunctor
import           Data.List
import           Data.List.Split
import           Data.String.Interpolate as S (i)
import           Debug.Trace
import           Layers
import           Mnist
import           Network.HTTP.Client
import           Numeric.LinearAlgebra

learnForward :: NElement a => ForwardNN a -> TrainBatch a -> (BackwardNN a, a)
learnForward (ForwardNN layers loss) (TrainBatch (t, x)) =
  result `seq` (BackwardNN layers' loss', result)
  where
    (layers', y) = forward layers x
    (loss', result) = output loss t y

learnBackward :: NElement a => a -> BackwardNN a -> ForwardNN a
learnBackward rate (BackwardNN layers loss) = ForwardNN layers' loss'
  where
    (loss', d) = backput loss
    (layers', _) = backward rate layers d

learn :: NElement a => a -> ForwardNN a -> TrainBatch a -> (ForwardNN a, a)
learn rate a = first (learnBackward rate) . learnForward a

learnAll ::
     NElement a => a -> ForwardNN a -> [TrainBatch a] -> (ForwardNN a, [a])
learnAll rate origin batches = ls `seq` (nn, ls)
  where
    (nn, ls) = foldr f (origin, []) batches
    f batch (a, ls) = ls `seq` second (~: ls) $ learn rate a batch
    x ~: xs = trace [i|[#{length xs + 1}/#{n}] #{show x}|] (x : xs)
    n = length batches

predict :: NElement a => ForwardLayer a -> Vector a -> Int
predict layers = maxIndex . flatten . snd . forward layers . asRow

evaluate :: NElement a => ForwardLayer a -> [(Int, Vector a)] -> Double
evaluate layers samples = fromIntegral nOk / fromIntegral (length samples)
  where
    nOk = length $ filter (uncurry (==)) results
    results = map (second $ predict layers) samples

normalAffine :: Int -> Int -> IO (ForwardLayer R)
normalAffine nIn nOut = do
  weights <- rand nIn nOut
  bias <- flatten <$> rand nOut 1
  return $ AffineForward weights bias

initNN :: ForwardLayer R -> [Int] -> IO (ForwardNN R)
initNN eoa ns = do
  (lastAffine:affines) <- mapM (uncurry normalAffine) $ spans [] ns
  let layers = foldl' (\b a -> a ~> eoa ~> b) lastAffine affines
  return $ ForwardNN layers SoftmaxWithCrossForward
  where
    spans rs [a, b]   = (a, b) : rs
    spans rs (a:b:xs) = spans ((a, b) : rs) (b : xs)

convertTrains :: Int -> MnistData -> [TrainBatch R]
convertTrains batchSize (MnistData src) =
  map (mkTrainer . unzip) $ chunksOf batchSize vectors
  where
    vectors = map (bimap (hotone 10) flatten) src
    mkTrainer (a, b) = TrainBatch (fromRows a, fromZ (fromRows b) / 255)

hotone :: (Integral v, NElement a) => Int -> v -> Vector a
hotone n' v = fromList $ map fromIntegral list
  where
    n =
      if i < n'
        then n'
        else error ("Too large value: " ++ show i)
    list = replicate i 0 ++ 1 : replicate (n - i - 1) 0
    i = fromIntegral v

convertTests :: MnistData -> [(Int, Vector R)]
convertTests (MnistData src) =
  map (bimap fromIntegral $ (/ 255) . flatten . fromZ) src

trainingSimple ::
     Double
  -> Int
  -> Int
  -> ForwardNN R
  -> MnistData
  -> MnistData
  -> ([Double], Double)
trainingSimple rate batchSize nReplicate nn trainData testData =
  (losses, evaluate layers $ convertTests testData)
  where
    trainBatches = convertTrains batchSize trainData
    batches = concat $ replicate nReplicate trainBatches
    (ForwardNN layers _, losses) = learnAll rate nn batches
