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
     * @param array{sampleSize?: int} $params
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

        $sampleSize = $params['sampleSize'] ?? 100;

        $this->estimator = new KMedoids(k: 3, sampleSize: $sampleSize, epochs: 50);
    }

    /**
     * Return the sample sizes to benchmark against.
     *
     * @return Generator<string, array{sampleSize: int}>
     */
    public function provideSampleSizes() : Generator
    {
        yield 'sample_size_50' => ['sampleSize' => 50];
        yield 'sample_size_100' => ['sampleSize' => 100];
        yield 'sample_size_250' => ['sampleSize' => 250];
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     * @ParamProviders("provideSampleSizes")
     *
     * @param array{sampleSize: int} $params
     */
    public function trainPredict(array $params) : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
