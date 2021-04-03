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
     * @test
     * @dataProvider determineProvider
     *
     * @param mixed $value
     * @param \Rubix\ML\DataType $expected
     */
    public function determine($value, DataType $expected) : void
    {
        $this->assertEquals($expected, DataType::detect($value));
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
        $value = imagecreatefrompng('tests/test.png');

        $this->assertEquals(DataType::image(), DataType::detect($value));
    }

    /**
     * @test
     * @dataProvider codeProvider
     *
     * @param \Rubix\ML\DataType $type
     * @param int $expected
     */
    public function code(DataType $type, int $expected) : void
    {
        $this->assertSame($expected, $type->code());
    }

    /**
     * @return \Generator<array>
     */
    public function codeProvider() : Generator
    {
        yield [DataType::categorical(), DataType::CATEGORICAL];

        yield [DataType::continuous(), DataType::CONTINUOUS];

        yield [DataType::image(), DataType::IMAGE];

        yield [DataType::other(), DataType::OTHER];
    }

    /**
     * @test
     */
    public function isCategorical() : void
    {
        $this->assertFalse(DataType::continuous()->isCategorical());
    }

    /**
     * @test
     */
    public function isContinuous() : void
    {
        $this->assertTrue(DataType::continuous()->isContinuous());
    }

    /**
     * @test
     */
    public function isImage() : void
    {
        $this->assertFalse(DataType::continuous()->isImage());
    }

    /**
     * @test
     */
    public function isOther() : void
    {
        $this->assertFalse(DataType::continuous()->isOther());
    }

    /**
     * @test
     */
    public function testToString() : void
    {
        $this->assertEquals('continuous', (string) DataType::continuous());
    }
}
