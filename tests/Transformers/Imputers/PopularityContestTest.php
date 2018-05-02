<?php

use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\PopularityContest;
use PHPUnit\Framework\TestCase;

class PopularityContestTest extends TestCase
{
    protected $values;

    protected $imputer;

    public function setUp()
    {
        $this->values = ['a', 'a', 'b', 'a', 'c'];

        $this->imputer = new PopularityContest();
    }

    public function test_build_local_celebrity_imputer()
    {
        $this->assertInstanceOf(PopularityContest::class, $this->imputer);
        $this->assertInstanceOf(Imputer::class, $this->imputer);
    }

    public function test_guess_value()
    {
        $this->imputer->fit($this->values);

        $value = $this->imputer->impute();

        $this->assertContains($value, $this->values);
    }
}
