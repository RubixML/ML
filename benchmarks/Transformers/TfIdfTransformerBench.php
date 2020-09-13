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
    protected const DATASET_SIZE = 10000;

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
        $mask = Matrix::rand(self::DATASET_SIZE, 4)
            ->greater(0.8);

        $samples = Matrix::gaussian(self::DATASET_SIZE, 4)
            ->multiply($mask)
            ->asArray();

        $this->dataset = Unlabeled::quick($samples);

        $this->transformer = new TfIdfTransformer(1.0);
    }

    /**
     * @Subject
     * @Iterations(3)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
