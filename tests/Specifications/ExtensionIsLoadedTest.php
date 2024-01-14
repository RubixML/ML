<?php

namespace Rubix\ML\Tests\Specifications;

use Rubix\ML\Specifications\ExtensionIsLoaded;
use PHPUnit\Framework\TestCase;
use Generator;

/**
 * @group Specifications
 * @requires extension json
 * @covers \Rubix\ML\Specifications\ExtensionIsLoaded
 */
class ExtensionIsLoadedTest extends TestCase
{
    /**
     * @test
     * @dataProvider passesProvider
     *
     * @param ExtensionIsLoaded $specification
     * @param bool $expected
     */
    public function passes(ExtensionIsLoaded $specification, bool $expected) : void
    {
        $this->assertSame($expected, $specification->passes());
    }

    /**
     * @return \Generator<mixed[]>
     */
    public function passesProvider() : Generator
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
}
