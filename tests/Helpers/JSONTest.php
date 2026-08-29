<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Helpers;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Helpers\JSON;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Exceptions\JSONException;
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

    public function testEncodeInvalidUTF8() : void
    {
        $this->expectException(JSONException::class);
        $this->expectExceptionMessage('Malformed UTF-8 characters, check encoding.');

        JSON::encode(['class' => "caf\xE9"]);
    }

    public function testDecodeNonArrayJson() : void
    {
        $this->expectException(JSONException::class);

        JSON::decode('42');
    }

    public function testDecodeBadData() : void
    {
        $this->expectException(RuntimeException::class);

        JSON::decode(data: '[{"package":...}]');
    }
}
