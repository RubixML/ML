<?php

namespace Rubix\ML\Tests\Other\Helpers;

use Rubix\ML\DataType;
use PHPUnit\Framework\TestCase;

class DataTypeTest extends TestCase
{
    public function test_determine_type() : void
    {
        $this->assertEquals(DataType::CATEGORICAL, DataType::determine('string'));
        $this->assertEquals(DataType::CONTINUOUS, DataType::determine(2.75));
        $this->assertEquals(DataType::CONTINUOUS, DataType::determine(-100));
        $this->assertEquals(DataType::OTHER, DataType::determine(null));
    }

    public function test_is_categorical() : void
    {
        $this->assertTrue(DataType::isCategorical('string'));
        $this->assertFalse(DataType::isCategorical(5.0));
        $this->assertFalse(DataType::isCategorical(3));
        $this->assertFalse(DataType::isCategorical(null));
    }

    public function test_is_continuous() : void
    {
        $this->assertFalse(DataType::isContinuous('string'));
        $this->assertTrue(DataType::isContinuous(5.0));
        $this->assertTrue(DataType::isContinuous(3));
        $this->assertFalse(DataType::isContinuous(null));
    }
    
    public function test_is_other() : void
    {
        $this->assertFalse(DataType::isOther('string'));
        $this->assertFalse(DataType::isOther(5.0));
        $this->assertFalse(DataType::isOther(3));
        $this->assertTrue(DataType::isOther(null));
    }
}
