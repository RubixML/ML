<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Specifications\ExtensionMaximumVersion;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @requires extension json
 * @covers \Rubix\ML\Specifications\ExtensionMaximumVersion
 */
class ExtensionMaximumVersionTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param ExtensionMaximumVersion $specification
     * @param bool $expected
     */
    public function passes(ExtensionMaximumVersion $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return Generator<mixed[]>
     */
    public function passesProvider() : Generator
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
}
