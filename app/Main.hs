{-# LANGUAGE QuasiQuotes #-}

module Main where

import           Control.Lens
import           Data.List
import           Data.String.Interpolate as S (i)
import           Layers
import           Learn
import           Mnist
import           System.Directory
import           System.Environment
import           Util

main :: IO ()
main = do
  args <- getArgs
  let dirPath =
        case args of
          path:_ -> path
          _      -> "."
  setCurrentDirectory dirPath
  doLearn

doLearn :: IO ()
doLearn =
  newManager defaultManagerSettings >>=
  saveMnist
    ".mnist"
    "http://yann.lecun.com/exdb/mnist/"
    [ "train-images-idx3-ubyte.gz"
    , "train-labels-idx1-ubyte.gz"
    , "t10k-images-idx3-ubyte.gz"
    , "t10k-labels-idx1-ubyte.gz"
    ] >>=
  mapM readMnist >>= \ms -> do
    let [MnistData srcTrains, MnistData srcTests] =
          map (\[Right a, Left b] -> b `zipMnist` a) $ chunksOf 2 ms
    trainers <- MnistData <$> shuffleList (take 60000 srcTrains)
    tests <- MnistData <$> shuffleList (take 10000 srcTests)
    origin <- initNN genAfterLayer [28 * 28, 50, 10]
    let (losses, result) = trainingSimple 0.1 1000 1000 origin trainers tests
    timestamp "Start training"
    timestamp [i|result=#{result * 100}%|]
    saveCSV ".lean_result.csv" ["index", "loss"] $
      map (^.. each) $ [0 ..] `zip` reverse losses
  where
    genAfterLayer n = do
      bn <- randomBatchNorm n
      return $ bn ~> ReLUForward
