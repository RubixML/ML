<?php

namespace Rubix\ML\Benchmarks\Clusterers;

use Rubix\ML\Clusterers\KMedoids;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Labeled;

use Generator;

/**
 * @Groups({"Clusterers"})
 * @BeforeMethods({"setUp"})
 */
class KMedoidsBench
{
    protected const TRAINING_SIZE = 10000;

    protected const TESTING_SIZE = 10000;

    /**
     * @var Labeled
     */
    protected Labeled $training;

    /**
     * @var Labeled
     */
    protected Labeled $testing;

    /**
     * @var KMedoids
     */
    protected KMedoids $estimator;

    /**
     * @param array{batchSize?: int} $params
     */
    public function setUp(array $params = []) : void
    {
        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $batchSize = $params['batchSize'] ?? 100;

        $this->estimator = new KMedoids(k: 3, batchSize: $batchSize, epochs: 10);
    }

    /**
     * Return the batch sizes to benchmark against.
     *
     * @return Generator<string, array{batchSize: int}>
     */
    public function provideSampleSizes() : Generator
    {
        yield 'batch size 50' => ['batchSize' => 50];

        yield 'batch size 100' => ['batchSize' => 100];

        yield 'batch size 250' => ['batchSize' => 250];
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     * @ParamProviders("provideSampleSizes")
     *
     * @param array{batchSize: int} $params
     */
    public function trainPredict(array $params) : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
