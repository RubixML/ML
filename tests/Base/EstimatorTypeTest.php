<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\EstimatorType;
use PHPUnit\Framework\TestCase;

#[Group('Other')]
#[CoversClass(EstimatorType::class)]
class EstimatorTypeTest extends TestCase
{
    protected EstimatorType $type;

    protected function setUp() : void
    {
        $this->type = new EstimatorType(EstimatorType::CLUSTERER);
    }

    public function testCode() : void
    {
        $this->assertSame(EstimatorType::CLUSTERER, $this->type->code());
    }

    public function testIsClassifier() : void
    {
        $this->assertFalse($this->type->isClassifier());
    }

    public function testIsRegressor() : void
    {
        $this->assertFalse($this->type->isRegressor());
    }

    public function testIsClusterer() : void
    {
        $this->assertTrue($this->type->isClusterer());
    }

    public function testIsAnomalyDetector() : void
    {
        $this->assertFalse($this->type->isAnomalyDetector());
    }

    public function testToString() : void
    {
        $this->assertEquals('clusterer', (string) $this->type);
    }
}
