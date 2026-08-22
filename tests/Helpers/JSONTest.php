<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Helpers')]
#[CoversClass(JSON::class)]
class JSONTest extends TestCase
{
    public function testDecode() : void
    {
        $actual = JSON::decode(data: '{"attitude":"nice","texture":"furry","sociability":"friendly","rating":4,"class":"not monster"}');

        $expected = [
            'attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster',
        ];

        $this->assertSame($expected, $actual);
    }

    public function testEncode() : void
    {
        $actual = JSON::encode(value: ['package' => 'rubix/ml']);

        $expected = '{"package":"rubix\/ml"}';

        $this->assertSame($expected, $actual);
    }

    public function testDecodeBadData() : void
    {
        $this->expectException(RuntimeException::class);

        JSON::decode(data: '[{"package":...}]');
    }
}
