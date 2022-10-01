<?php

namespace Rubix\ML\Benchmarks\Kernels\Distance;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Kernels\Distance\SafeEuclidean;
use Rubix\ML\Transformers\LambdaFunction;

/**
 * @Groups({"DistanceKernels"})
 * @BeforeMethods({"setUp"})
 */
class SafeEuclideanBench
{
    protected const NUM_SAMPLES = 10000;

    /**
     * @var list<list<float>>
     */
    protected $aSamples;

    /**
     * @var list<list<float>>
     */
    protected $bSamples;

    /**
     * @var \Rubix\ML\Kernels\Distance\SafeEuclidean
     */
    protected $kernel;

    public function setUp() : void
    {
        $generator = new Blob([0, 0, 0, 0, 0, 0, 0, 0], 5.0);

        $dropValues = new LambdaFunction((function ($sample) {
            $sample[4] = rand(0, 5) === 0 ? NAN : $sample[4];
            $sample[5] = rand(0, 10) === 0 ? NAN : $sample[5];
        }));

        $this->aSamples = $generator->generate(self::NUM_SAMPLES)
            ->apply($dropValues)
            ->samples();

        $this->bSamples = $generator->generate(self::NUM_SAMPLES)
            ->apply($dropValues)
            ->samples();

        $this->kernel = new SafeEuclidean();
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
