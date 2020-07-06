<?php

namespace Rubix\ML\Benchmarks\Kernels\Distance;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\Euclidean;

/**
 * @Groups({"DistanceKernels"})
 * @BeforeMethods({"setUp"})
 */
class EuclideanBench
{
    protected const NUM_SAMPLES = 10000;

    /**
     * @var array[]
     */
    protected $aSamples;

    /**
     * @var array[]
     */
    protected $bSamples;

    /**
     * @var \Rubix\ML\Kernels\Distance\Euclidean
     */
    protected $kernel;

    public function setUp() : void
    {
        $generator = new Blob([0, 0, 0, 0, 0, 0, 0, 0], 5.0);

        $this->aSamples = $generator->generate(self::NUM_SAMPLES)->samples();
        $this->bSamples = $generator->generate(self::NUM_SAMPLES)->samples();

        $this->kernel = new Euclidean();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function compute() : void
    {
        array_map([$this->kernel, 'compute'], $this->aSamples, $this->bSamples);
    }
}
