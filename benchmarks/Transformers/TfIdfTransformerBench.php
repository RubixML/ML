<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Tensor\Matrix;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Transformers\TfIdfTransformer;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class TfIdfTransformerBench
{
    protected const NUM_SAMPLES = 10000;

    /**
     * @var \Rubix\ML\Datasets\Unlabeled
     */
    public $dataset;

    /**
     * @var \Rubix\ML\Transformers\TfIdfTransformer
     */
    protected $transformer;

    public function setUp() : void
    {
        $mask = Matrix::rand(self::NUM_SAMPLES, 100)
            ->greater(0.8);

        $samples = Matrix::gaussian(self::NUM_SAMPLES, 100)
            ->multiply($mask)
            ->asArray();

        $this->dataset = Unlabeled::quick($samples);

        $this->transformer = new TfIdfTransformer();
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("seconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
