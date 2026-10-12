<?php

namespace Rubix\ML\Benchmarks\Clusterers;

use Rubix\ML\Clusterers\OPTICS;
use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Datasets\Generators\Agglomerate;
use Rubix\ML\Datasets\Labeled;

/**
 * @Groups({"Clusterers"})
 * @BeforeMethods({"setUp"})
 */
class OPTICSBench
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
     * @var OPTICS
     */
    protected OPTICS $estimator;

    public function setUp() : void
    {
        $generator = new Agglomerate([
            'red' => new Blob([0.0, 0.0], [1.0, 1.0]),
            'green' => new Blob([10.0, 0.0], [1.0, 1.0]),
            'blue' => new Blob([5.0, 8.0], [1.0, 1.0]),
        ]);

        $this->training = $generator->generate(self::TRAINING_SIZE);

        $this->testing = $generator->generate(self::TESTING_SIZE);

        $this->estimator = new OPTICS(radius: 2.0, minDensity: 20);
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
