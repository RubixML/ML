<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Specifications\ExtensionMaximumVersion;
use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[CoversClass(ExtensionMaximumVersion::class)]
class ExtensionMaximumVersionTest extends TestCase
{
    /**
     * @return Generator<mixed[]>
     */
    public static function passesProvider() : Generator
    {
        yield [
            ExtensionMaximumVersion::with('json', '0.0.0'),
            false,
        ];

        yield [
            ExtensionMaximumVersion::with('json', '999.0.0'),
            true,
        ];

        yield [
            ExtensionMaximumVersion::with('What about the forest?', '999.0.0'),
            false,
        ];
    }

    /**
     * @param ExtensionMaximumVersion $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    #[Test]
    public function passes(ExtensionMaximumVersion $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
