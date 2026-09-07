<?php

namespace Rubix\ML\Benchmarks\Regressors;

use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Datasets\Generators\Hyperplane;

use Generator;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Backends\Backend;
use Rubix\ML\Specifications\ExtensionIsLoaded;

/**
 * @Groups({"Regressors"})
 * @BeforeMethods({"setUp"})
 */
class KNNRegressorBench
{
    protected const int TRAINING_SIZE = 10000;

    protected const int TESTING_SIZE = 10000;

    protected Labeled $training;

    protected Labeled $testing;

    protected KNNRegressor $estimator;

    /**
     * @return Generator<string, array{backend: Backend}>
     */
    public static function provideBackends() : Generator
    {
        $serialBackend = new Serial();

        yield (string) $serialBackend => [
            'backend' => $serialBackend,
        ];

        $ampBackend = new Amp();

        yield (string) $ampBackend => [
            'backend' => $ampBackend,
        ];

        if (ExtensionIsLoaded::with('swoole')->passes()) {
            $swooleBackend = new Swoole();

            yield (string) $swooleBackend => [
                'backend' => $swooleBackend,
            ];
        }
    }

    public function setUp() : void
    {
        $generator = new Hyperplane([1, 5.5, -7, 0.01], 0.0);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new KNNRegressor(5);
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     * @ParamProviders("provideBackends")
     * @param array{ backend: Backend } $params
     */
    public function trainPredict(array $params) : void
    {
        $this->estimator->setBackend($params['backend']);

        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
