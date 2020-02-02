<?php

namespace Rubix\ML\Tests;

use Rubix\ML\DataType;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Other
 * @covers \Rubix\ML\DataType
 */
class DataTypeTest extends TestCase
{
    /**
     * @var \Rubix\ML\DataType
     */
    protected $dataType;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->dataType = new DataType(DataType::CONTINUOUS);
    }

    /**
     * @test
     * @dataProvider determineProvider
     *
     * @param mixed $value
     * @param \Rubix\ML\DataType $expected
     */
    public function determine($value, DataType $expected) : void
    {
        $this->assertEquals($expected, DataType::determine($value));
    }

    /**
     * @return \Generator<array>
     */
    public function determineProvider() : Generator
    {
        yield ['string', DataType::categorical()];

        yield ['3', DataType::categorical()];

        yield [2.75, DataType::continuous()];

        yield [-100, DataType::continuous()];

        yield [null, DataType::other()];

        yield [false, DataType::other()];

        yield [[], DataType::other()];

        yield [(object) [], DataType::other()];
    }

    /**
     * @test
     * @requires extension gd
     */
    public function determineImage() : void
    {
        $value = imagecreatefromjpeg('tests/space.jpg');

        $this->assertEquals(DataType::image(), DataType::determine($value));
    }

    /**
     * @test
     */
    public function type() : void
    {
        $this->assertSame(DataType::CONTINUOUS, $this->dataType->type());
    }
    
    /**
     * @test
     */
    public function isCategorical() : void
    {
        $this->assertFalse($this->dataType->isCategorical());
    }
    
    /**
     * @test
     */
    public function isContinuous() : void
    {
        $this->assertTrue($this->dataType->isContinuous());
    }

    /**
     * @test
     */
    public function isImage() : void
    {
        $this->assertFalse($this->dataType->isImage());
    }
        
    /**
     * @test
     */
    public function isOther() : void
    {
        $this->assertFalse($this->dataType->isOther());
    }

    /**
     * @test
     */
    public function testToString() : void
    {
        $this->assertEquals('continuous', (string) $this->dataType);
    }
}
