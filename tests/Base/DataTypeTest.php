<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\DataType;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Other')]
#[CoversClass(DataType::class)]
class DataTypeTest extends TestCase
{
    public static function determineProvider() : Generator
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

    public static function codeProvider() : Generator
    {
        yield [DataType::categorical(), DataType::CATEGORICAL];

        yield [DataType::continuous(), DataType::CONTINUOUS];

        yield [DataType::image(), DataType::IMAGE];

        yield [DataType::other(), DataType::OTHER];
    }

    /**
     * @param mixed $value
     * @param DataType $expected
     */
    #[DataProvider('determineProvider')]
    #[Test]
    public function determine(mixed $value, DataType $expected) : void
    {
        $this->assertEquals($expected, DataType::detect($value));
    }

    #[RequiresPhpExtension('gd')]
    #[Test]
    public function determineImage() : void
    {
        $value = imagecreatefrompng('tests/test.png');

        $this->assertEquals(DataType::image(), DataType::detect($value));
    }

    /**
     * @param DataType $type
     * @param int $expected
     */
    #[DataProvider('codeProvider')]
    #[Test]
    public function code(DataType $type, int $expected) : void
    {
        $this->assertSame($expected, $type->code());
    }

    #[Test]
    public function isCategorical() : void
    {
        $this->assertFalse(DataType::continuous()->isCategorical());
    }

    #[Test]
    public function isContinuous() : void
    {
        $this->assertTrue(DataType::continuous()->isContinuous());
    }

    #[Test]
    public function isImage() : void
    {
        $this->assertFalse(DataType::continuous()->isImage());
    }

    #[Test]
    public function isOther() : void
    {
        $this->assertFalse(DataType::continuous()->isOther());
    }

    public function testToString() : void
    {
        $this->assertEquals('continuous', (string) DataType::continuous());
    }
}
