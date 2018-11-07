<?php

namespace Rubix\ML\Tests\Regressors;

use Rubix\ML\Online;
use Rubix\ML\Learner;
use Rubix\ML\Verbose;
use Rubix\ML\Estimator;
use Rubix\ML\Persistable;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\NeuralNet\Layers\Dense;
use Rubix\ML\Regressors\MLPRegressor;
use Rubix\ML\Other\Loggers\BlackHole;
use Rubix\ML\NeuralNet\Optimizers\Adam;
use Rubix\ML\NeuralNet\Layers\Activation;
use Rubix\ML\Datasets\Generators\SwissRoll;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\NeuralNet\CostFunctions\LeastSquares;
use Rubix\ML\NeuralNet\ActivationFunctions\LeakyReLU;
use Rubix\ML\CrossValidation\Metrics\MeanSquaredError;
use PHPUnit\Framework\TestCase;
use InvalidArgumentException;
use RuntimeException;

class MLPRegressorTest extends TestCase
{
    const TRAIN_SIZE = 400;
    const TEST_SIZE = 5;
    const TOLERANCE = 3;

    protected $generator;

    protected $estimator;

    public function setUp()
    {
        $this->generator = new SwissRoll(4., -7., 0., 1., 0.3);

        $this->estimator = new MLPRegressor([
            new Dense(10),
            new Activation(new LeakyReLU()),
            new Dense(10),
            new Activation(new LeakyReLU()),
        ], 10, new Adam(0.01), 1e-4, 100, 1e-3, new LeastSquares(), 0.1, new MeanSquaredError(), 3);

        $this->estimator->setLogger(new BlackHole());
    }

    public function test_build_regressor()
    {
        $this->assertInstanceOf(MLPRegressor::class, $this->estimator);
        $this->assertInstanceOf(Online::class, $this->estimator);
        $this->assertInstanceOf(Learner::class, $this->estimator);
        $this->assertInstanceOf(Verbose::class, $this->estimator);
        $this->assertInstanceOf(Persistable::class, $this->estimator);
        $this->assertInstanceOf(Estimator::class, $this->estimator);
    }

    public function test_estimator_type()
    {
        $this->assertEquals(Estimator::REGRESSOR, $this->estimator->type());
    }

    public function test_train_partial_predict_proba()
    {
        $dataset = $this->generator->generate(self::TRAIN_SIZE + self::TEST_SIZE);

        $transformer = new ZScaleStandardizer();

        $transformer->fit($dataset);
        $dataset->apply($transformer);

        $testing = $dataset->randomize()->take(self::TEST_SIZE);

        $folds = $dataset->fold(3);

        $this->estimator->train($folds[0]);
        $this->estimator->partial($folds[1]);
        $this->estimator->partial($folds[2]);

        foreach ($this->estimator->predict($testing) as $i => $prediction) {
            $this->assertEquals($testing->label($i), $prediction, '', self::TOLERANCE);
        }
    }

    public function test_train_with_unlabeled()
    {
        $this->expectException(InvalidArgumentException::class);

        $this->estimator->train(Unlabeled::quick());
    }

    public function test_predict_untrained()
    {
        $this->expectException(RuntimeException::class);

        $this->estimator->predict(Unlabeled::quick());
    }
}
