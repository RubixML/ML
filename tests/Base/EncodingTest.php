<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Encoding;
use PHPUnit\Framework\TestCase;

#[Group('Other')]
#[CoversClass(Encoding::class)]
class EncodingTest extends TestCase
{
    protected const array TEST_DATA = [
        'breakfast' => 'pancakes',
        'lunch' => 'croque monsieur',
        'dinner' => 'new york strip steak',
    ];

    protected Encoding $encoding;

    protected function setUp() : void
    {
        $this->encoding = new Encoding(json_encode(self::TEST_DATA) ?: '');
    }

    public function testData() : void
    {
        $expected = '{"breakfast":"pancakes","lunch":"croque monsieur","dinner":"new york strip steak"}';

        $this->assertEquals($expected, $this->encoding->data());
    }

    public function tstBytes() : void
    {
        $this->assertSame(82, $this->encoding->bytes());
    }
}
