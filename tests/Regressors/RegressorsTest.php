<?php

namespace Rubix\ML\Tests\Regressors;

use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\TestDox;
use PHPUnit\Framework\TestCase;
use Rubix\ML\CrossValidation\Metrics\RSquared;
use Rubix\ML\Datasets\Generators\Hyperplane;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\NeuralNet\CostFunctions\HuberLoss\HuberLoss;
use Rubix\ML\NeuralNet\Optimizers\Adam\Adam;
use Rubix\ML\NeuralNet\Optimizers\Adam as LegacyAdam;
use Rubix\ML\Regressors\Adaline as LegacyAdaline;
use Rubix\ML\Regressors\Adaline\Adaline as NDAdaline;
use Rubix\ML\Regressors\MLPRegressor as LegacyMLPRegressor;
use Rubix\ML\Regressors\MLPRegressor\MLPRegressor as NDMLPRegressor;

class RegressorsTest extends TestCase{

    private Labeled $dataset;

    protected function setUp() : void
    {
        // Data: [area, floor, distance to center, age of house]
        $samples = [
            [50, 3, 5, 10, 1],
            [70, 10, 3, 5, 2],
            [40, 2, 8, 30, 3],
        ];

        $targets = [
            66_000,
            95_000,
            45_000,
        ];

        // Create dataset
        $this->dataset = new Labeled($samples, $targets);
    }

//    #[Test]
//    #[TestDox('testAdaline')]
    public function runAdaline() {

        $regression = new NDAdaline(
            batchSize: $this->dataset->numSamples(),
            optimizer: new Adam(0.01),
            l2Penalty: 0.0,
            epochs: 5000,
            minChange: 1e-8,
            window: 50
        );

        $regression->train($this->dataset);

        $dataset = new Unlabeled($this->dataset->samples());
        $predictions = $regression->predict($dataset);

        $metric = new RSquared();
        $score = $metric->score($predictions, $this->dataset->labels());

        self::assertGreaterThan(0.8, $score);

    }

//    #[Test]
//    #[TestDox('testAdalineLegacy')]
    public function runAdalineLegacy() {

        $regression = new LegacyAdaline(
            batchSize: $this->dataset->numSamples(),
            l2Penalty: 0.0,
            epochs: 5000,
            minChange: 1e-8,
            window: 50
        );

        $regression->train($this->dataset);

        $dataset = new Unlabeled($this->dataset->samples());
        $predictions = $regression->predict($dataset);

        $metric = new RSquared();
        $score = $metric->score($predictions, $this->dataset->labels());

        self::assertGreaterThan(0.99, $score);
    }

//    #[Test]
//    #[TestDox('testMLPRegressor')]
    public function runMLPRegressor() {

        srand(0);

        $regression = new NDMLPRegressor(
            hiddenLayers: [],
            batchSize: $this->dataset->numSamples(),
            optimizer: new Adam(0.001),
            epochs: 10000,
            minChange: 1e-8,
            window: 50,
            holdOut: 0.0
        );

        $regression->train($this->dataset);

        $dataset = new Unlabeled($this->dataset->samples());
        $predictions = $regression->predict($dataset);

        $metric = new RSquared();
        $score = $metric->score($predictions, $this->dataset->labels());

        self::assertGreaterThan(0.8, $score);

    }

//    #[Test]
//    #[TestDox('testMLPRegressorLegacy')]
    public function runMLPRegressorLegacy() {

        srand(0);

        $regression = new LegacyMLPRegressor(
            hiddenLayers: [],
            batchSize: $this->dataset->numSamples(),
            optimizer: new LegacyAdam(0.001),
            epochs: 10000,
            minChange: 1e-8,
            window: 50,
            holdOut: 0.0
        );

        $regression->train($this->dataset);

        $dataset = new Unlabeled($this->dataset->samples());
        $predictions = $regression->predict($dataset);

        $metric = new RSquared();
        $score = $metric->score($predictions, $this->dataset->labels());

        self::assertGreaterThan(0.8, $score);

    }

    #[Test]
    /**
     * Test method ...
     * @return void
     */
    public function test() {
        self::assertTrue(true);
    }


}
