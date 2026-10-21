<?php

namespace Rubix\ML\Benchmarks\Base;

use Rubix\ML\GeneticSearch;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Kernels\Distance\Euclidean;

/**
 * Genetic Search Benchmark
 *
 * @Groups({"Base"})
 * @BeforeMethods({"setUp"})
 */
class GeneticSearchBench
{
    /**
     * The number of samples used to train each candidate.
     */
    protected const int TRAINING_SIZE = 200;

    /**
     * The number of samples used for inference.
     */
    protected const int TESTING_SIZE = 200;

    /**
     * The training dataset.
     *
     * @var Labeled
     */
    protected Labeled $training;

    /**
     * The testing dataset.
     *
     * @var Labeled
     */
    protected Labeled $testing;

    /**
     * The genetic search estimator.
     *
     * @var GeneticSearch
     */
    protected GeneticSearch $estimator;

    /**
     * Generate datasets and initialize a search with three generations.
     */
    public function setUp() : void
    {
        srand(42);

        $generator = new Agglomerate([
            'Iris-setosa' => new Blob([5.0, 3.42, 1.46, 0.24], [0.35, 0.38, 0.17, 0.1]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
        ]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new GeneticSearch(
            class: KNearestNeighbors::class,
            params: [
                [1, 3, 5, 7, 9, 11],
                [true, false],
                [new Euclidean()],
            ],
            populationSize: 4,
            generations: 3,
        );
    }

    /**
     * Measure candidate search, final training, and inference.
     *
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function trainPredict() : void
    {
        $this->estimator->train($this->training);

        $this->estimator->predict($this->testing);
    }
}
