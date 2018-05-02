<?php

use Rubix\Engine\Transformers\Imputers\Imputer;
use Rubix\Engine\Transformers\Imputers\BlurryMean;
use PHPUnit\Framework\TestCase;

class BlurryMeanTest extends TestCase
{
    protected $values;

    protected $imputer;

    public function setUp()
    {
        $this->values = [1, 2, 3, 4, 5];

        $this->imputer = new BlurryMean();
    }

    public function test_build_blurry_mean_imputer()
    {
        $this->assertInstanceOf(BlurryMean::class, $this->imputer);
        $this->assertInstanceOf(Imputer::class, $this->imputer);
    }

    public function test_guess_value()
    {
        $this->imputer->fit($this->values);

        $value = $this->imputer->impute();

        $this->assertThat($value,$this->logicalAnd($this->greaterThan(2.5), $this->lessThan(3.5)));
    }
}
