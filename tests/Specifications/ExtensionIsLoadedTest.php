<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Specifications;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\DataProvider;
use PHPUnit\Framework\Attributes\Group;
use PHPUnit\Framework\Attributes\RequiresPhpExtension;
use Rubix\ML\Specifications\ExtensionIsLoaded;
use PHPUnit\Framework\TestCase;
use Generator;

#[Group('Specifications')]
#[RequiresPhpExtension('json')]
#[CoversClass(ExtensionIsLoaded::class)]
class ExtensionIsLoadedTest extends TestCase
{
    public static function passesProvider() : Generator
    {
        yield [
            ExtensionIsLoaded::with('json'),
            true,
        ];

        yield [
            ExtensionIsLoaded::with("I be trappin' where I go"),
            false,
        ];
    }

    /**
     * @param ExtensionIsLoaded $specification
     * @param bool $expected
     */
    #[DataProvider('passesProvider')]
    public function testPasses(ExtensionIsLoaded $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }
}
