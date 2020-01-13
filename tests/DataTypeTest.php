<?php

namespace Rubix\ML\Tests;

use Rubix\ML\DataType;
use PHPUnit\Framework\TestCase;

/**
 * @group DataTypes
 * @covers \Rubix\ML\DataType
 */
class DataTypeTest extends TestCase
{
    /**
     * @test
     */
    public function determine() : void
    {
        $this->assertEquals(DataType::CATEGORICAL, DataType::determine('string'));
        $this->assertEquals(DataType::CONTINUOUS, DataType::determine(2.75));
        $this->assertEquals(DataType::CONTINUOUS, DataType::determine(-100));
        $this->assertEquals(DataType::IMAGE, DataType::determine(imagecreatefromjpeg('tests/space.jpg')));
        $this->assertEquals(DataType::OTHER, DataType::determine(null));
    }
    
    /**
     * @test
     */
    public function isCategorical() : void
    {
        $this->assertTrue(DataType::isCategorical('string'));
        $this->assertFalse(DataType::isCategorical(5.0));
        $this->assertFalse(DataType::isCategorical(3));
        $this->assertFalse(DataType::isCategorical(imagecreatefromjpeg('tests/space.jpg')));
        $this->assertFalse(DataType::isCategorical(null));
    }
    
    /**
     * @test
     */
    public function isContinuous() : void
    {
        $this->assertFalse(DataType::isContinuous('string'));
        $this->assertTrue(DataType::isContinuous(5.0));
        $this->assertTrue(DataType::isContinuous(3));
        $this->assertFalse(DataType::isContinuous(imagecreatefromjpeg('tests/space.jpg')));
        $this->assertFalse(DataType::isContinuous(null));
    }

    /**
     * @test
     */
    public function isImage() : void
    {
        $this->assertFalse(DataType::isImage('string'));
        $this->assertFalse(DataType::isImage(5.0));
        $this->assertFalse(DataType::isImage(3));
        $this->assertTrue(DataType::isImage(imagecreatefromjpeg('tests/space.jpg')));
        $this->assertFalse(DataType::isImage(null));
    }
        
    /**
     * @test
     */
    public function isOther() : void
    {
        $this->assertFalse(DataType::isOther('string'));
        $this->assertFalse(DataType::isOther(5.0));
        $this->assertFalse(DataType::isOther(3));
        $this->assertFalse(DataType::isOther(imagecreatefromjpeg('tests/space.jpg')));
        $this->assertTrue(DataType::isOther(null));
    }
}
