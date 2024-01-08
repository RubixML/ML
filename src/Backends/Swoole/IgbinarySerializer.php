<?php

namespace Rubix\ML\Backends\Swoole;

class IgbinarySerializer implements Serializer
{
    public function serialize(mixed $value) : string
    {
        return igbinary_serialize($value);
    }

    public function unserialize(string $data) : mixed
    {
        return igbinary_unserialize($data);
    }
}
