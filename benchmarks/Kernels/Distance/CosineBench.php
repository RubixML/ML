<?php

namespace Rubix\ML\Benchmarks\Kernels\Distance;

use Tensor\Matrix;
use Rubix\ML\Kernels\Distance\Cosine;

/**
 * @Groups({"DistanceKernels"})
 */
class CosineBench
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
     * @var Cosine
     */
    protected $kernel;

    public function setUp() : void
    {
        $this->kernel = new Cosine();
    }

    public function setUpDense() : void
    {
        $this->aSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)->asArray();
        $this->bSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)->asArray();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @BeforeMethods({"setUp", "setUpDense"})
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function computeDense() : void
    {
        array_map([$this->kernel, 'compute'], $this->aSamples, $this->bSamples);
    }

    public function setUpSparse() : void
    {
        $mask = Matrix::rand(self::NUM_SAMPLES, 8)
            ->greater(0.5);

        $this->aSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)
            ->multiply($mask)
            ->asArray();

        $mask = Matrix::rand(self::NUM_SAMPLES, 8)
            ->greater(0.5);

        $this->bSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)
            ->multiply($mask)
            ->asArray();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @BeforeMethods({"setUp", "setUpSparse"})
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function computeSparse() : void
    {
        array_map([$this->kernel, 'compute'], $this->aSamples, $this->bSamples);
    }

    public function setUpVerySparse() : void
    {
        $mask = Matrix::rand(self::NUM_SAMPLES, 8)
            ->greater(0.9);

        $this->aSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)
            ->multiply($mask)
            ->asArray();

        $mask = Matrix::rand(self::NUM_SAMPLES, 8)
            ->greater(0.9);

        $this->bSamples = Matrix::gaussian(self::NUM_SAMPLES, 8)
            ->multiply($mask)
            ->asArray();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @BeforeMethods({"setUp", "setUpVerySparse"})
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function computeVerySparse() : void
    {
        array_map([$this->kernel, 'compute'], $this->aSamples, $this->bSamples);
    }
}
