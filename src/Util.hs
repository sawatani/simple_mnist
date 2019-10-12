{-# LANGUAGE ScopedTypeVariables #-}

module Util (
  shuffleList
, saveCSV
, timestamp
) where

import           Control.DeepSeq
import           Data.ByteString.Lazy        as BS hiding (map)
import           Data.Csv
import           Data.Time.Format
import           Data.Time.LocalTime
import qualified Data.Vector                 as V
import qualified Data.Vector.Generic         as VG
import qualified Data.Vector.Generic.Mutable as VM
import           System.Random.MWC

shuffleList :: [a] -> IO [a]
shuffleList [a] = return [a]
shuffleList xs =  withSystemRandom . asGenST $ \g -> do
    v <- VG.unsafeThaw $ VG.fromList xs
    repeatSwap v g n
    (v' :: V.Vector a) <- VG.unsafeFreeze v
    return $ VG.toList v'
    where
        n = Prelude.length xs - 1
        repeatSwap v g i = if (i < 0) then return () else do
            j <- uniformR (0, n) g
            VM.swap v j i
            repeatSwap v g (i - 1)

saveCSV :: Show a => FilePath -> [String] -> [[a]] -> IO ()
saveCSV filePath columnNames body = BS.writeFile filePath buf
    where
      buf = encode $ columnNames : rows
      rows = map (map show) body

timestamp :: String -> IO ()
timestamp msg = msg `deepseq` do
    t <- getZonedTime
    let fm = "[%Y/%m/%d %H:%M:%S] " ++ msg
    print $ formatTime defaultTimeLocale fm t
