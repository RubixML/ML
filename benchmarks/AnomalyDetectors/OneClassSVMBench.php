<?php

namespace Rubix\ML\Benchmarks\AnomalyDetectors;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\AnomalyDetectors\OneClassSVM;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Labeled;

/**
 * @Groups({"AnomalyDetectors"})
 * @BeforeMethods({"setUp"})
 */
class OneClassSVMBench
{
    protected const int TRAINING_SIZE = 10000;

    protected const int TESTING_SIZE = 10000;

    protected Labeled $training;

    protected Labeled $testing;

    protected OneClassSVM $estimator;

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'Iris-virginica' => new Blob([6.59, 2.97, 5.55, 2.03], [0.63, 0.32, 0.55, 0.27]),
            'Iris-versicolor' => new Blob([5.94, 2.77, 4.26, 1.33], [0.51, 0.31, 0.47, 0.2]),
        ], [0.99, 0.01]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new OneClassSVM();
    }

    /**
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
