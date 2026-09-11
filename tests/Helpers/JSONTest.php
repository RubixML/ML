<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\JSONException;
use PHPUnit\Framework\TestCase;

#[Group('Helpers')]
#[CoversClass(JSON::class)]
class JSONTest extends TestCase
{
    #[Test]
    public function decode() : void
    {
        $actual = JSON::decode(data: '{"attitude":"nice","texture":"furry","sociability":"friendly","rating":4,"class":"not monster"}');

        $expected = [
            'attitude' => 'nice', 'texture' => 'furry', 'sociability' => 'friendly', 'rating' => 4, 'class' => 'not monster',
        ];

        $this->assertSame($expected, $actual);
    }

    #[Test]
    public function encode() : void
    {
        $actual = JSON::encode(value: ['package' => 'rubix/ml']);

        $expected = '{"package":"rubix\/ml"}';

        $this->assertSame($expected, $actual);
    }

    #[Test]
    public function encodeInvalidUTF8() : void
    {
        $this->expectException(JSONException::class);
        $this->expectExceptionMessage('Malformed UTF-8 characters, check encoding.');

        JSON::encode(['class' => "caf\xE9"]);
    }

    #[Test]
    public function decodeNonArrayJson() : void
    {
        $this->expectException(JSONException::class);

        JSON::decode('42');
    }

    #[Test]
    public function decodeBadData() : void
    {
        $this->expectException(RuntimeException::class);

        JSON::decode(data: '[{"package":...}]');
    }
}
