//Copyright (c) 2006 Bruno van Dooren
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.


using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace FunctionPointerProxy
{
  public class AdvancedProxy : IProxy
  {
    [UnmanagedFunctionPointerAttribute(CallingConvention.Cdecl)]
    public delegate int AdvancedProxyDelegate(
      //A simple I4 parameter. Nothing special
      [MarshalAs(UnmanagedType.I4)] int i,
      //An ASCII string. this will be marshalled as UNICODE on the
      //.NET side and ASCII on the unmanaged side.
      [MarshalAs(UnmanagedType.LPStr)] string aString,
      //A UNICODE string. This will be marshalled as UNICODE on the
      //.NET side, and UNICODE on the unmanaged side.
      [MarshalAs(UnmanagedType.LPWStr)] string wString);
    
    //export an event because that is the only way we are going to get
    //LabVIEW to give use a delegate.
    public event AdvancedProxyDelegate AdvancedEvent;

    //extract the delegate from the event. If there was not yet an event
    //registered, this will trigger an exception which is caught by the
    //LabVIEW .NET interface node. No need to worry about it here.
    public Delegate GetDelegate()
    {
      return AdvancedEvent.GetInvocationList()[0];
    }
  }
}
