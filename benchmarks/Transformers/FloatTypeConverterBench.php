<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\FloatTypeConverter;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class FloatTypeConverterBench
{
    protected const DATASET_SIZE = 100000;

    /**
     * @var \Rubix\ML\Datasets\Dataset
     */
    public $dataset;

    /**
     * @var FloatTypeConverter
     */
    protected $transformer;

    public function setUp() : void
    {
        $generator = new Blob([0.0, 0.0, 0.0, 0.0]);

        $this->dataset = $generator->generate(self::DATASET_SIZE)
            ->apply(new LambdaFunction(function (&$sample) {
                $sample[0] = (int) $sample[0];
                $sample[2] = (int) $sample[2];
            }));

        $this->transformer = new FloatTypeConverter();
    }

    /**
     * @Subject
     * @Iterations(5)
     * @OutputTimeUnit("milliseconds", precision=3)
     */
    public function apply() : void
    {
        $this->dataset->apply($this->transformer);
    }
}
