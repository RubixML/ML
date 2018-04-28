<?php

use Rubix\Engine\Transformers\Strategies\Strategy;
use Rubix\Engine\Transformers\Strategies\BlurryMean;
use PHPUnit\Framework\TestCase;

class BlurryMeanTest extends TestCase
{
    protected $strategy;

    public function setUp()
    {
        $this->strategy = new BlurryMean();
    }

    public function test_build_blurry_mean_strategy()
    {
        $this->assertInstanceOf(BlurryMean::class, $this->strategy);
        $this->assertInstanceOf(Strategy::class, $this->strategy);
    }

    public function test_guess_value()
    {
        $data = [1, 2, 3, 4];

        $value = $this->strategy->guess($data);

        $this->assertTrue($value < 3 && $value > 2);
    }
}
