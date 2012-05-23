<?xml version='1.0'?>
<Project Type="Project" LVVersion="8208000">
   <Item Name="My Computer" Type="My Computer">
      <Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
      <Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
      <Property Name="server.tcp.enabled" Type="Bool">false</Property>
      <Property Name="server.tcp.port" Type="Int">0</Property>
      <Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
      <Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
      <Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
      <Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
      <Property Name="specify.custom.address" Type="Bool">false</Property>
      <Item Name="Subvi" Type="Folder">
         <Item Name="CleanupCallback.vi" Type="VI" URL="CleanupCallback.vi"/>
         <Item Name="GetDelegateFunctionPointer.vi" Type="VI" URL="GetDelegateFunctionPointer.vi"/>
         <Item Name="SimpleCallBack.vi" Type="VI" URL="SimpleCallBack.vi"/>
         <Item Name="t_CallbackReferences.ctl" Type="VI" URL="t_CallbackReferences.ctl"/>
         <Item Name="AdvancedCallback.vi" Type="VI" URL="AdvancedCallback.vi"/>
      </Item>
      <Item Name="SimpleDemo.vi" Type="VI" URL="SimpleDemo.vi"/>
      <Item Name="AdvancedDemo.vi" Type="VI" URL="AdvancedDemo.vi"/>
      <Item Name="Dependencies" Type="Dependencies"/>
      <Item Name="Build Specifications" Type="Build"/>
   </Item>
</Project>
