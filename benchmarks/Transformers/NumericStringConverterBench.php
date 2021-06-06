<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Generators\Blob;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\NumericStringConverter;

/**
 * @Groups({"Transformers"})
 * @BeforeMethods({"setUp"})
 */
class NumericStringConverterBench
{
    protected const DATASET_SIZE = 100000;

    /**
     * @var \Rubix\ML\Datasets\Dataset
     */
    public $dataset;

    /**
     * @var \Rubix\ML\Transformers\NumericStringConverter
     */
    protected $transformer;

    public function setUp() : void
    {
        $generator = new Blob([0.0, 0.0, 0.0, 0.0]);

        $this->dataset = $generator->generate(self::DATASET_SIZE)
            ->apply(new LambdaFunction(function (&$sample) {
                $sample[1] = strval($sample[1]);
                $sample[3] = strval($sample[3]);
            }));

        $this->transformer = new NumericStringConverter();
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
