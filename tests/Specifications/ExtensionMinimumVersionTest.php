<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Specifications\ExtensionMinimumVersion;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[RequiresPhpExtension('json')]
#[CoversClass(ExtensionMinimumVersion::class)]
class ExtensionMinimumVersionTest extends TestCase
{
    public static function passesProvider() : Generator
    {
        yield [
            ExtensionMinimumVersion::with('json', '0.0.0'),
            true,
        ];

        yield [
            ExtensionMinimumVersion::with('json', '999.0.0'),
            false,
        ];

        yield [
            ExtensionMinimumVersion::with('What about the forest?', '0.0.0'),
            false,
        ];
    }

    /**
     * @param ExtensionMinimumVersion $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(ExtensionMinimumVersion $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
