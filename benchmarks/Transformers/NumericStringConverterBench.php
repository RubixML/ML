<?php

namespace Rubix\ML\Benchmarks\Transformers;

use Rubix\ML\Datasets\Generators\Blob;
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

        $this->dataset = $generator->generate(self::DATASET_SIZE)->map(function ($record) {
            $record[1] = strval($record[1]);
            $record[3] = strval($record[3]);

            return $record;
        });

        $this->transformer = new NumericStringConverter();
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
