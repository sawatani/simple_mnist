{-# LANGUAGE FlexibleContexts #-}

module Learn
  ( newManager
  , defaultManagerSettings
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
import           Debug.Trace
import           Layers
import           Mnist
import           Network.HTTP.Client
import           Numeric.LinearAlgebra

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
hotone n v = fromList $ map fromIntegral list
  where
    zero = if i < n then 0 else error ("Too large value: " ++ show i)
    list = replicate i zero ++ 1 : replicate (n - i - 1) zero
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
