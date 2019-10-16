module SynapseSpec (spec) where

import           Numeric.LinearAlgebra as A
import           Synapse
import           Test.Hspec
import           Test.Hspec.QuickCheck (prop)
import           Test.QuickCheck
import           Text.Printf

runTest :: IO ()
runTest = hspec spec

spec :: Spec
spec = do
    describe "Sigmoid" $ do
        prop "make with exp" $
            forAll anySignal $ \n ->
                sigmoid n `shouldBe` (1.0 / (1.0 + exp (-n)))
    describe "Sigmoid Matrix" $ do
        prop "apply all elements" $
            forAll (choose (1, 10)) $ \m ->
                forAll (choose (1, 10)) $ \n ->
                    forAll (vectorOf (m * n) anySignal) $ \vs ->
                        let r = sigmoidm $ (m A.>< n) vs
                            rs = A.toList `concatMap` A.toRows r
                            xs = sigmoid `map` vs
                         in rs `shouldBe` xs
    describe "Sigmoid Backward" $ do
        prop "calc diff" $
            forAll (choose (1, 10)) $ \m ->
                forAll (choose (1, 10)) $ \n ->
                    forAll (vectorOf (m * n) anySignal) $ \ys ->
                        forAll (vectorOf (m * n) anySignal) $ \ds ->
                            let r =
                                    sigmoidBackward
                                        ((m A.>< n) ys)
                                        ((m A.>< n) ds)
                                rs = A.toList `concatMap` A.toRows r
                                xs =
                                    (\(y, d) -> d * (1.0 - y) * y) `map`
                                    (ys `zip` ds)
                             in rs `shouldBe` xs

anySignal :: Gen Double
anySignal = (arbitrary :: Gen Double) `suchThat` (\a -> -10 <= a && a <= 10)
