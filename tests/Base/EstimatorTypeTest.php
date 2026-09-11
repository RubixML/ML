<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
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

    #[Test]
    public function code() : void
    {
        $this->assertSame(EstimatorType::CLUSTERER, $this->type->code());
    }

    #[Test]
    public function isClassifier() : void
    {
        $this->assertFalse($this->type->isClassifier());
    }

    #[Test]
    public function isRegressor() : void
    {
        $this->assertFalse($this->type->isRegressor());
    }

    #[Test]
    public function isClusterer() : void
    {
        $this->assertTrue($this->type->isClusterer());
    }

    #[Test]
    public function isAnomalyDetector() : void
    {
        $this->assertFalse($this->type->isAnomalyDetector());
    }

    #[Test]
    public function testToString() : void
    {
        $this->assertEquals('clusterer', (string) $this->type);
    }
}
