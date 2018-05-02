<?php

use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\KMostFrequent;
use PHPUnit\Framework\TestCase;

class KMostFrequentTest extends TestCase
{
    protected $values;

    protected $imputer;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'b', 'c'];

        $this->imputer = new KMostFrequent(2);
    }

    public function test_build_k_most_frequent_imputer()
    {
        $this->assertInstanceOf(KMostFrequent::class, $this->imputer);
        $this->assertInstanceOf(Imputer::class, $this->imputer);
    }

    public function test_guess_value()
    {
        $this->imputer->fit($this->values);

        $value = $this->imputer->impute();

        $this->assertContains($value, $this->values);
    }
}
