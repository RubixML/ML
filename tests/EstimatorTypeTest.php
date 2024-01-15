<?php

namespace Rubix\ML\Tests;

use Rubix\ML\EstimatorType;
use PHPUnit\Framework\TestCase;

/**
 * @group Other
 * @covers \Rubix\ML\EstimatorType
 */
class EstimatorTypeTest extends TestCase
{
    /**
     * @var EstimatorType
     */
    protected $type;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->type = new EstimatorType(EstimatorType::CLUSTERER);
    }

    /**
     * @test
     */
    public function code() : void
    {
        $this->assertSame(EstimatorType::CLUSTERER, $this->type->code());
    }

    /**
     * @test
     */
    public function isClassifier() : void
    {
        $this->assertFalse($this->type->isClassifier());
    }

    /**
     * @test
     */
    public function isRegressor() : void
    {
        $this->assertFalse($this->type->isRegressor());
    }

    /**
     * @test
     */
    public function isClusterer() : void
    {
        $this->assertTrue($this->type->isClusterer());
    }

    /**
     * @test
     */
    public function isAnomalyDetector() : void
    {
        $this->assertFalse($this->type->isAnomalyDetector());
    }

    /**
     * @test
     */
    public function testToString() : void
    {
        $this->assertEquals('clusterer', (string) $this->type);
    }
}
