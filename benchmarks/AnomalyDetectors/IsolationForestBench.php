<?php

namespace Rubix\ML\Benchmarks\AnomalyDetectors;

use Rubix\ML\Backends\Backend;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\AnomalyDetectors\IsolationForest;
use Rubix\ML\Datasets\Labeled;
use Generator;
use Rubix\ML\Backends\Serial;
use Rubix\ML\Backends\Amp;
use Rubix\ML\Backends\Swoole;
use Rubix\ML\Specifications\SpecificationChain;
use Rubix\ML\Specifications\ExtensionIsLoaded;

/**
 * @Groups({"AnomalyDetectors"})
 * @BeforeMethods({"setUp"})
 */
class IsolationForestBench
{
    protected const int TRAINING_SIZE = 10000;

    protected const int TESTING_SIZE = 10000;

    protected Labeled $training;

    protected Labeled $testing;

    protected IsolationForest $estimator;

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

        if (
            SpecificationChain::with([
                new ExtensionIsLoaded('swoole'),
                new ExtensionIsLoaded('igbinary'),
            ])->passes()
        ) {
            $swooleBackend = new Swoole();

            yield (string) $swooleBackend => [
                'backend' => $swooleBackend,
            ];
        }
    }

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
        ], [0.99, 0.01]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new IsolationForest();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @ParamProviders("provideBackends")
     * @OutputTimeUnit("seconds", precision=3)
     * @param array{ backend: Backend } $params
     */
    public function trainPredict(array $params) : void
    {
        $this->estimator->setBackend($params['backend']);

        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
